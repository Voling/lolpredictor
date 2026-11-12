import hashlib
import hmac
import random
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
import jwt

from ..config import Settings

ICONS = tuple(range(29))
DAY = 86_400
LINK = "link"
REQUEST = re.compile(r"^[A-Za-z0-9_=-]{16,128}$")
TIMEOUT = 5.0
KEYS_REFRESH = 600.0
KEYS_RETRY = 10.0
PENDING_SECONDS = 900
RECENT = "recent"
RECENT_LIMIT = 5
RECENT_DAYS = 30
CHECK_SECONDS = 10
RIOT_BUDGET = "riot#budget"
CHECKS_PER_MINUTE = 30
VISITOR = "visitor"
VISITOR_LABEL = b"lolpredictor visitor v1"
LOCAL_VISITOR_KEY = b"lolpredictor local visitor key"
ANONYMOUS = "anonymous"
EVERYONE_FULL = "Checks without an account are full for today. Sign in to keep checking."
RIOT_BUSY = "Busy, try again in a minute."
RIOT_DOWN = "Riot isn't answering us right now. Try again later."
RIOT_REFUSED = "Riot refused that request. Check the name and tag."
UNREACHABLE = "We couldn't reach Riot. Try again later."
CONFLICT_TRIES = 3
CHARGED, DUPLICATE, FULL = "charged", "duplicate", "full"


class AuthError(Exception):
    pass


class QuotaExceeded(Exception):
    pass


class LinkError(Exception):
    pass


class RateLimited(LinkError):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__("Riot is busy right now. Try again in a few minutes.")


class RiotUnavailable(RateLimited):
    def __init__(self, retry_after: float):
        LinkError.__init__(self, RIOT_DOWN)
        self.retry_after = retry_after


class RiotRefused(LinkError):
    pass


class NotFound(RiotRefused):
    pass


class Busy(Exception):
    pass


class RiotBusy(Busy):
    pass


class SigningKeys:
    def __init__(self, fetch, refresh: float = KEYS_REFRESH, retry: float = KEYS_RETRY, clock=time.monotonic):
        self.fetch, self.refresh, self.retry, self.clock = fetch, refresh, retry, clock
        self.keys: dict = {}
        self.loaded: float | None = None
        self.failed: float | None = None

    def _due(self) -> bool:
        now = self.clock()
        if self.failed is not None:
            return now - self.failed >= self.retry
        return self.loaded is None or now - self.loaded >= self.refresh

    def __call__(self, token: str):
        kid = jwt.get_unverified_header(token).get("kid")
        if kid not in self.keys and self._due():
            try:
                found = {item["kid"]: jwt.PyJWK(item).key for item in self.fetch().get("keys", []) if "kid" in item}
            except Exception:
                self.failed = self.clock()
                raise
            if not found:
                self.failed = self.clock()
                raise jwt.InvalidKeyError("the pool published no signing keys")
            self.keys, self.loaded, self.failed = found, self.clock(), None
        if kid not in self.keys:
            raise jwt.InvalidKeyError("unknown signing key")
        return self.keys[kid]


class TokenVerifier:
    def __init__(self, issuer: str, keys):
        self.issuer, self.keys = issuer, keys

    def subject(self, token: str | None) -> str:
        if not token:
            raise AuthError("Sign in first.")
        try:
            key = self.keys(token)
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthError("We couldn't check your sign in. Try again in a moment.") from exc
        except jwt.PyJWTError as exc:
            raise AuthError("Your sign in expired. Sign in again.") from exc
        try:
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer, options={"require": ["exp", "iss", "sub", "token_use"]})
        except jwt.PyJWTError as exc:
            raise AuthError("Your sign in expired. Sign in again.") from exc
        if claims["token_use"] != "access":
            raise AuthError("Sign in again.")
        return claims["sub"]


def _published_keys(url: str) -> dict:
    response = httpx.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def cognito_verifier(settings: Settings) -> TokenVerifier:
    issuer = f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/{settings.cognito_pool_id}"
    return TokenVerifier(issuer, SigningKeys(lambda: _published_keys(f"{issuer}/.well-known/jwks.json")))


def request_key(kind: str, day: str, *parts) -> str:
    text = "|".join([kind, day, *(" ".join(str(part or "").lower().split()) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def _value(item: dict):
    kind, value = next(iter(item.items()))
    if kind == "N":
        number = float(value)
        return int(number) if number.is_integer() else number
    if kind == "BOOL":
        return bool(value)
    return value


def _typed(value) -> dict:
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, (int, float)):
        return {"N": str(value)}
    return {"S": str(value)}


def _item(values: dict) -> dict:
    return {name: _typed(value) for name, value in values.items() if value is not None}


class DynamoTable:
    def __init__(self, name: str, client=None, pause=time.sleep):
        if client is None:
            import boto3

            client = boto3.client("dynamodb")
        self.name, self.client, self.pause = name, client, pause

    def _key(self, pk: str, sk: str) -> dict:
        return {"pk": {"S": pk}, "sk": {"S": sk}}

    def _counter(self, pk: str, sk: str, amount: int, limit: int, expires: int) -> dict:
        return {
            "TableName": self.name,
            "Key": self._key(pk, sk),
            "UpdateExpression": "ADD used :amount SET expires = :expires",
            "ConditionExpression": "attribute_not_exists(used) OR used <= :room",
            "ExpressionAttributeValues": {
                ":amount": {"N": str(amount)},
                ":room": {"N": str(limit - amount)},
                ":expires": {"N": str(expires)},
            },
        }

    def _retry(self, call):
        for attempt in range(CONFLICT_TRIES):
            try:
                return call()
            except self.client.exceptions.TransactionConflictException:
                self.pause(random.uniform(0.02, 0.1) * (attempt + 1))
        raise Busy("Busy right now. Try again in a moment.")

    def get(self, pk: str, sk: str) -> dict | None:
        found = self.client.get_item(TableName=self.name, Key=self._key(pk, sk), ConsistentRead=True).get("Item")
        return {name: _value(value) for name, value in found.items()} if found else None

    def put(self, pk: str, sk: str, values: dict) -> None:
        self._retry(lambda: self.client.put_item(TableName=self.name, Item={**self._key(pk, sk), **_item(values)}))

    def delete(self, pk: str, sk: str) -> None:
        self._retry(lambda: self.client.delete_item(TableName=self.name, Key=self._key(pk, sk)))

    def put_if(self, pk: str, sk: str, values: dict, match: dict) -> bool:
        names = {f"#m{index}": name for index, name in enumerate(match)}
        checks = ["attribute_exists(pk)", *(f"attribute_not_exists(#m{index})" if value is None else f"#m{index} = :m{index}" for index, value in enumerate(match.values()))]
        request = {"TableName": self.name, "Item": {**self._key(pk, sk), **_item(values)}, "ConditionExpression": " AND ".join(checks), "ExpressionAttributeNames": names}
        typed = {f":m{index}": _typed(value) for index, value in enumerate(match.values()) if value is not None}
        if typed:
            request["ExpressionAttributeValues"] = typed

        def call():
            try:
                self.client.put_item(**request)
            except self.client.exceptions.ConditionalCheckFailedException:
                return False
            return True

        return self._retry(call)

    def every(self, pk: str) -> list[dict]:
        request = {"TableName": self.name, "KeyConditionExpression": "pk = :pk", "ExpressionAttributeValues": {":pk": {"S": pk}}}
        items = []
        while True:
            found = self.client.query(**request)
            items += [{name: _value(value) for name, value in item.items()} for item in found.get("Items", [])]
            if not found.get("LastEvaluatedKey"):
                return items
            request["ExclusiveStartKey"] = found["LastEvaluatedKey"]

    def latest(self, pk: str, limit: int, fields: tuple[str, ...] | None = None) -> list[dict]:
        request = {
            "TableName": self.name,
            "KeyConditionExpression": "pk = :pk",
            "ExpressionAttributeValues": {":pk": {"S": pk}},
            "ScanIndexForward": False,
            "Limit": limit,
        }
        if fields:
            request["ProjectionExpression"] = ", ".join(["sk", *(f"#f{index}" for index in range(len(fields)))])
            request["ExpressionAttributeNames"] = {f"#f{index}": name for index, name in enumerate(fields)}
        found = self.client.query(**request)
        return [{name: _value(value) for name, value in item.items()} for item in found.get("Items", [])]

    def claim(self, pk: str, sk: str, owner: str) -> bool:
        try:
            self.client.put_item(
                TableName=self.name,
                Item={**self._key(pk, sk), "owner": {"S": owner}},
                ConditionExpression="attribute_not_exists(pk) OR #owner = :owner",
                ExpressionAttributeNames={"#owner": "owner"},
                ExpressionAttributeValues={":owner": {"S": owner}},
            )
        except self.client.exceptions.ConditionalCheckFailedException:
            return False
        return True

    def reserve(self, pk: str, sk: str, owner: str, now: int, expires: int) -> bool:
        try:
            self.client.put_item(
                TableName=self.name,
                Item={**self._key(pk, sk), "owner": {"S": owner}, "expires": {"N": str(expires)}},
                ConditionExpression="attribute_not_exists(pk) OR #owner = :owner OR expires < :now",
                ExpressionAttributeNames={"#owner": "owner"},
                ExpressionAttributeValues={":owner": {"S": owner}, ":now": {"N": str(now)}},
            )
        except self.client.exceptions.ConditionalCheckFailedException:
            return False
        return True

    def release(self, pk: str, sk: str, owner: str) -> None:
        try:
            self.client.delete_item(
                TableName=self.name,
                Key=self._key(pk, sk),
                ConditionExpression="#owner = :owner",
                ExpressionAttributeNames={"#owner": "owner"},
                ExpressionAttributeValues={":owner": {"S": owner}},
            )
        except self.client.exceptions.ConditionalCheckFailedException:
            pass

    def add(self, pk: str, sk: str, amount: int, limit: int, expires: int) -> int | None:
        def call():
            try:
                found = self.client.update_item(**self._counter(pk, sk, amount, limit, expires), ReturnValues="UPDATED_NEW")
            except self.client.exceptions.ConditionalCheckFailedException:
                return None
            return int(found["Attributes"]["used"]["N"])

        return self._retry(call)

    @staticmethod
    def _matching(match: dict) -> tuple[str, dict, dict]:
        names = {f"#m{index}": name for index, name in enumerate(match)}
        values = {f":m{index}": _typed(value) for index, value in enumerate(match.values())}
        return " AND ".join(f"#m{index} = :m{index}" for index in range(len(match))), names, values

    def _update(self, pk: str, sk: str, expression: str, condition: str, names: dict, values: dict, returns: str = "NONE") -> dict | None:
        request = {"TableName": self.name, "Key": self._key(pk, sk), "UpdateExpression": expression, "ExpressionAttributeNames": names, "ExpressionAttributeValues": values, "ReturnValues": returns}
        if condition:
            request["ConditionExpression"] = condition

        def call():
            try:
                return self.client.update_item(**request)
            except self.client.exceptions.ConditionalCheckFailedException:
                return None

        return self._retry(call)

    def collect(self, pk: str, sk: str, fields: list[str], member: str, match: dict) -> dict | None:
        condition, names, values = self._matching(match)
        names = {**names, **{f"#f{index}": field for index, field in enumerate(fields)}}
        expression = "ADD " + ", ".join(f"#f{index} :member" for index in range(len(fields)))
        found = self._update(pk, sk, expression, condition, names, {**values, ":member": {"SS": [member]}}, "ALL_NEW")
        return None if found is None else {name: _value(value) for name, value in found["Attributes"].items()}

    def add_to_window(self, pk: str, sk: str, amount: int, limit: int, start: float) -> bool:
        values = {":amount": _typed(amount), ":start": _typed(start), ":room": _typed(limit - amount)}
        return self._update(pk, sk, "ADD used :amount", "#start = :start AND used <= :room", {"#start": "start"}, values) is not None

    def start_window(self, pk: str, sk: str, amount: int, start: float, old: float | None, expires: int) -> bool:
        request = {
            "TableName": self.name,
            "Item": {**self._key(pk, sk), **_item({"used": amount, "start": start, "expires": expires})},
            "ConditionExpression": "attribute_not_exists(#start)" if old is None else "#start = :old",
            "ExpressionAttributeNames": {"#start": "start"},
        }
        if old is not None:
            request["ExpressionAttributeValues"] = {":old": _typed(old)}

        def call():
            try:
                self.client.put_item(**request)
            except self.client.exceptions.ConditionalCheckFailedException:
                return False
            return True

        return self._retry(call)

    def flag(self, pk: str, sk: str, field: str, value: str, unless: str | None = None) -> bool:
        names = {"#f": field, **({"#u": unless} if unless else {})}
        condition = "attribute_exists(pk) AND (attribute_not_exists(#f) OR #f = :v)" + (" AND attribute_not_exists(#u)" if unless else "")
        return self._update(pk, sk, "SET #f = :v", condition, names, {":v": _typed(value)}) is not None

    def raise_to(self, pk: str, sk: str, field: str, value, match: dict | None = None, expires: int | None = None, also: dict | None = None) -> bool:
        matched, names, values = self._matching(match or {})
        also = also or {}
        expression = "SET #f = :v" + (", expires = :expires" if expires is not None else "") + "".join(f", #a{index} = :a{index}" for index in range(len(also)))
        condition = "(attribute_not_exists(#f) OR #f < :v)" + (f" AND {matched}" if matched else "")
        values = {**values, ":v": _typed(value), **({":expires": _typed(int(expires))} if expires is not None else {}), **{f":a{index}": _typed(extra) for index, extra in enumerate(also.values())}}
        names = {**names, "#f": field, **{f"#a{index}": name for index, name in enumerate(also)}}
        return self._update(pk, sk, expression, condition, names, values) is not None

    def charge(self, pk: str, counter: str, request: str, amount: int, limit: int, expires: int) -> str:
        for attempt in range(CONFLICT_TRIES):
            try:
                self.client.transact_write_items(
                    TransactItems=[
                        {
                            "Put": {
                                "TableName": self.name,
                                "Item": {**self._key(pk, request), "expires": {"N": str(expires)}},
                                "ConditionExpression": "attribute_not_exists(pk)",
                            }
                        },
                        {"Update": self._counter(pk, counter, amount, limit, expires)},
                    ]
                )
            except self.client.exceptions.TransactionCanceledException as exc:
                reasons = [reason.get("Code") for reason in exc.response.get("CancellationReasons", [])]
                if reasons[:1] == ["ConditionalCheckFailed"]:
                    return DUPLICATE
                if reasons[1:2] == ["ConditionalCheckFailed"]:
                    return FULL
                if "TransactionConflict" not in reasons:
                    raise
                self.pause(random.uniform(0.02, 0.1) * (attempt + 1))
                continue
            return CHARGED
        raise Busy("Busy right now. Try again in a moment.")


class LocalTable:
    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _copy(found: dict) -> dict:
        return {name: sorted(value) if isinstance(value, set) else value for name, value in found.items()}

    @staticmethod
    def _matches(found: dict | None, match: dict) -> bool:
        return found is not None and all(found.get(name) == value for name, value in match.items())

    def get(self, pk: str, sk: str) -> dict | None:
        with self.lock:
            found = self.items.get((pk, sk))
            return None if found is None else self._copy(found)

    def put(self, pk: str, sk: str, values: dict) -> None:
        with self.lock:
            self.items[(pk, sk)] = {name: value for name, value in values.items() if value is not None}

    def delete(self, pk: str, sk: str) -> None:
        with self.lock:
            self.items.pop((pk, sk), None)

    def put_if(self, pk: str, sk: str, values: dict, match: dict) -> bool:
        with self.lock:
            found = self.items.get((pk, sk))
            if not self._matches(found, match):
                return False
            self.items[(pk, sk)] = {name: value for name, value in values.items() if value is not None}
            return True

    def every(self, pk: str) -> list[dict]:
        with self.lock:
            return [{"pk": pk, "sk": sk, **self._copy(found)} for (part, sk), found in sorted(self.items.items()) if part == pk]

    def add(self, pk: str, sk: str, amount: int, limit: int, expires: int) -> int | None:
        with self.lock:
            found = self.items.get((pk, sk), {})
            if "used" in found and found["used"] > limit - amount:
                return None
            self.items[(pk, sk)] = {**found, "used": found.get("used", 0) + amount, "expires": expires}
            return self.items[(pk, sk)]["used"]

    def collect(self, pk: str, sk: str, fields: list[str], member: str, match: dict) -> dict | None:
        with self.lock:
            found = self.items.get((pk, sk))
            if not self._matches(found, match):
                return None
            for field in fields:
                found[field] = {*found.get(field, ()), member}
            return self._copy(found)

    def add_to_window(self, pk: str, sk: str, amount: int, limit: int, start: float) -> bool:
        with self.lock:
            found = self.items.get((pk, sk))
            if found is None or found.get("start") != start or found.get("used", 0) > limit - amount:
                return False
            found["used"] = found.get("used", 0) + amount
            return True

    def start_window(self, pk: str, sk: str, amount: int, start: float, old: float | None, expires: int) -> bool:
        with self.lock:
            found = self.items.get((pk, sk)) or {}
            if found.get("start") != old:
                return False
            self.items[(pk, sk)] = {"used": amount, "start": start, "expires": expires}
            return True

    def flag(self, pk: str, sk: str, field: str, value: str, unless: str | None = None) -> bool:
        with self.lock:
            found = self.items.get((pk, sk))
            if found is None or found.get(field, value) != value or (unless and unless in found):
                return False
            found[field] = value
            return True

    def raise_to(self, pk: str, sk: str, field: str, value, match: dict | None = None, expires: int | None = None, also: dict | None = None) -> bool:
        with self.lock:
            found = self.items.get((pk, sk))
            if match and not self._matches(found, match):
                return False
            if found is not None and field in found and found[field] >= value:
                return False
            self.items[(pk, sk)] = {**(found or {}), field: value, **({"expires": int(expires)} if expires is not None else {}), **(also or {})}
            return True


class RiotAccounts:
    def __init__(self, key, platform: str, region: str, limiter, client: httpx.Client | None = None):
        self.key, self.limiter, self.client = key, limiter, client or httpx.Client(timeout=TIMEOUT)
        self.platform, self.region = platform, region
        self.account_host = f"https://{region}.api.riotgames.com"
        self.platform_host = f"https://{platform}.api.riotgames.com"

    def _ask(self, route: str, *urls: str) -> list:
        key = self.key()
        held = self.limiter.held(key)
        if held:
            raise RiotUnavailable(held)
        wait = self.limiter.take(route, len(urls))
        if wait:
            raise RateLimited(wait)
        return [self._get(route, url, key) for url in urls]

    def _get(self, route: str, url: str, key: str):
        try:
            response = self.client.get(url, headers={"X-Riot-Token": key})
        except httpx.HTTPError as exc:
            raise LinkError(UNREACHABLE) from exc
        status = response.status_code
        if status in (401, 403):
            held = self.limiter.trip(key, status)
            getattr(self.key, "forget", lambda: None)()
            raise RiotUnavailable(held)
        if status == 404:
            raise NotFound("Riot doesn't know that Riot ID. Check the name and tag.")
        if status == 429:
            retry_after = float(response.headers.get("Retry-After") or 10.0)
            self.limiter.cool(route, retry_after)
            raise RateLimited(retry_after)
        if 400 <= status < 500:
            raise RiotRefused(RIOT_REFUSED)
        if status != 200:
            raise LinkError(UNREACHABLE)
        return response.json()

    def _match_url(self, match_id: str) -> str:
        return f"{self.account_host}/lol/match/v5/matches/{quote(match_id, safe='')}"

    def account(self, riot_id: str) -> dict:
        name, _, tag = riot_id.partition("#")
        return self._ask(self.region, f"{self.account_host}/riot/account/v1/accounts/by-riot-id/{quote(name, safe='')}/{quote(tag, safe='')}")[0]

    def account_by_puuid(self, puuid: str) -> dict:
        return self._ask(self.region, f"{self.account_host}/riot/account/v1/accounts/by-puuid/{quote(puuid, safe='')}")[0]

    def league(self, puuid: str) -> list[dict]:
        return list(self._ask(self.platform, f"{self.platform_host}/lol/league/v4/entries/by-puuid/{quote(puuid, safe='')}")[0])

    def icon(self, puuid: str) -> int:
        return int(self._ask(self.platform, f"{self.platform_host}/lol/summoner/v4/summoners/by-puuid/{quote(puuid, safe='')}")[0]["profileIconId"])

    def match_ids(self, puuid: str, kind: str, count: int = 100) -> list[str]:
        return list(self._ask(self.region, f"{self.account_host}/lol/match/v5/matches/by-puuid/{quote(puuid, safe='')}/ids?type={kind}&count={count}")[0])

    def match(self, match_id: str) -> dict:
        return self._ask(self.region, self._match_url(match_id))[0]

    def timeline(self, match_id: str) -> dict:
        return self._ask(self.region, f"{self._match_url(match_id)}/timeline")[0]

    def game(self, match_id: str) -> tuple[dict, dict]:
        match, timeline = self._ask(self.region, self._match_url(match_id), f"{self._match_url(match_id)}/timeline")
        return match, timeline


def ssm_key(name: str):
    found = {}

    def key() -> str:
        if "value" not in found:
            import boto3

            found["value"] = boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
        return found["value"]

    key.forget = found.clear
    return key


def riot_id_of(text: str) -> str:
    riot_id = " ".join(str(text or "").split())
    name, mark, tag = riot_id.partition("#")
    if not mark or not name.strip() or not tag.strip() or len(riot_id) > 40:
        raise LinkError("Enter your Riot ID as name#tag.")
    return f"{name.strip()}#{tag.strip()}"


def _shuffled(items: list) -> list:
    return random.SystemRandom().sample(items, len(items))


class Accounts:
    def __init__(
        self,
        table,
        riot,
        daily: int,
        attempts: int,
        clock=time.time,
        shuffle=_shuffled,
        visitor_duos: int = 20,
        anonymous_duos: int = 3000,
        visitor_riot_budget: int = 20,
        checks_per_minute: int = CHECKS_PER_MINUTE,
    ):
        self.table, self.riot, self.daily, self.attempts = table, riot, daily, attempts
        self.visitor_duos, self.anonymous_duos, self.visitor_riot_budget = visitor_duos, anonymous_duos, visitor_riot_budget
        self.clock, self.shuffle, self.checks_per_minute = clock, shuffle, checks_per_minute

    def _user(self, user: str) -> str:
        return f"user#{user}"

    def day(self) -> str:
        return f"{datetime.fromtimestamp(self.clock(), timezone.utc):%Y-%m-%d}"

    def _day(self, kind: str) -> tuple[str, int]:
        return f"{kind}#{self.day()}", int(self.clock()) + 2 * DAY

    def visitor_riot_calls(self, calls: int) -> bool:
        now = int(self.clock())
        if calls > self.visitor_riot_budget or self.table.add(RIOT_BUDGET, f"visitors#{now // 60}", calls, self.visitor_riot_budget, now + DAY) is None:
            raise RiotBusy(RIOT_BUSY)
        return True

    def _attempt(self, user: str) -> None:
        key, expires = self._day("links")
        if self.table.add(self._user(user), key, 1, self.attempts, expires) is None:
            raise QuotaExceeded(f"You can try linking {self.attempts} times a day. Try again tomorrow.")

    def _give_back_attempt(self, user: str) -> None:
        key, expires = self._day("links")
        self.table.add(self._user(user), key, -1, self.attempts + 1, expires)

    def link(self, user: str) -> dict:
        return self.table.get(self._user(user), LINK) or {}

    def _pending(self, link: dict) -> dict | None:
        if not link.get("pending_puuid") or int(self.clock()) - int(link.get("pending_started", 0)) > PENDING_SECONDS:
            return None
        return {"riot_id": link["pending_riot_id"], "puuid": link["pending_puuid"], "icon": link["pending_icon"]}

    def _verified(self, link: dict) -> dict:
        return {name: link[name] for name in ("riot_id", "puuid", "verified") if name in link and link.get("verified")}

    def status(self, user: str) -> dict:
        link = self.link(user)
        pending = self._pending(link)
        key, _ = self._day("duos")
        used = int((self.table.get(self._user(user), key) or {}).get("used", 0))
        return {
            "riot_id": link.get("riot_id") if link.get("verified") else None,
            "verified": bool(link.get("verified", False)),
            "pending": {"riot_id": pending["riot_id"], "icon": pending["icon"]} if pending else None,
            "daily": self.daily,
            "remaining": max(0, self.daily - used),
        }

    def start_link(self, user: str, text: str) -> dict:
        riot_id = riot_id_of(text)
        link = self.link(user)
        pending = self._pending(link)
        if pending and pending["riot_id"].lower() == riot_id.lower():
            return self.status(user)
        self._attempt(user)
        try:
            account = self.riot.account(riot_id)
            current = self.riot.icon(account["puuid"])
        except RateLimited:
            self._give_back_attempt(user)
            raise
        puuid = account["puuid"]
        now = int(self.clock())
        icon = next(
            (choice for choice in self.shuffle([c for c in ICONS if c != current]) if self.table.reserve(f"puuid#{puuid}", f"pending#{choice}", user, now, now + PENDING_SECONDS)),
            None,
        )
        if icon is None:
            raise LinkError("Too many people are linking that Riot account right now. Try again in 15 minutes.")
        if link.get("pending_puuid") and (link["pending_puuid"], link["pending_icon"]) != (puuid, icon):
            self.table.release(f"puuid#{link['pending_puuid']}", f"pending#{link['pending_icon']}", user)
        name = f"{account.get('gameName', riot_id.partition('#')[0])}#{account.get('tagLine', riot_id.partition('#')[2])}"
        self.table.put(
            self._user(user),
            LINK,
            {**self._verified(link), "pending_riot_id": name, "pending_puuid": puuid, "pending_icon": icon, "pending_started": now},
        )
        return self.status(user)

    def verify_link(self, user: str) -> dict:
        link = self.link(user)
        if not link.get("pending_puuid"):
            raise LinkError("Start by entering your Riot ID.")
        pending = self._pending(link)
        if pending is None:
            self.table.release(f"puuid#{link['pending_puuid']}", f"pending#{link['pending_icon']}", user)
            self.table.put(self._user(user), LINK, self._verified(link))
            raise LinkError("That link expired. Enter your Riot ID again.")
        now = int(self.clock())
        if self.table.add(self._user(user), f"check#{now // CHECK_SECONDS}", 1, 1, now + DAY) is None:
            return self.status(user)
        if self.table.add(RIOT_BUDGET, f"checks#{now // 60}", 1, self.checks_per_minute, now + DAY) is None:
            return self.status(user)
        try:
            icon = self.riot.icon(pending["puuid"])
        except RateLimited:
            return self.status(user)
        if icon != pending["icon"]:
            return self.status(user)
        if not self.table.claim(f"puuid#{pending['puuid']}", "owner", user):
            raise LinkError("That Riot account is already linked to another lolpredictor account.")
        if link.get("verified") and link.get("puuid") != pending["puuid"]:
            self.table.release(f"puuid#{link['puuid']}", "owner", user)
        self.table.release(f"puuid#{pending['puuid']}", f"pending#{pending['icon']}", user)
        self.table.put(self._user(user), LINK, {"riot_id": pending["riot_id"], "puuid": pending["puuid"], "verified": True})
        return self.status(user)

    def remember(self, duos: list[dict]) -> None:
        now = int(self.clock() * 1000)
        for index, duo in enumerate(duos):
            kept = {name: value for name, value in duo.items() if value is not None}
            self.table.put(RECENT, f"{now:014d}#{index}#{random.SystemRandom().randrange(1 << 20):06x}", {**kept, "at": now // 1000, "expires": now // 1000 + RECENT_DAYS * DAY})

    def recent(self, limit: int = RECENT_LIMIT) -> list[dict]:
        return [{name: value for name, value in item.items() if name not in ("pk", "sk", "expires")} for item in self.table.latest(RECENT, limit)]

    def linked_puuid(self, user: str) -> str | None:
        link = self.link(user)
        return link["puuid"] if link.get("verified") else None

    def _full(self, user: str) -> QuotaExceeded:
        return QuotaExceeded(f"You have {self.status(user)['remaining']} of {self.daily} duo checks left today.")

    def spend(self, user: str, count: int, request: str | None = None) -> tuple[int, bool]:
        if count > self.daily:
            raise QuotaExceeded(f"You can check {self.daily} duos a day.")
        key, expires = self._day("duos")
        if request and REQUEST.match(request):
            outcome = self.table.charge(self._user(user), key, f"request#{request}", count, self.daily, expires)
            if outcome == FULL:
                raise self._full(user)
            return self.status(user)["remaining"], outcome == CHARGED
        used = self.table.add(self._user(user), key, count, self.daily, expires)
        if used is None:
            raise self._full(user)
        return self.daily - used, True

    def refund(self, user: str, count: int, request: str | None = None) -> None:
        key, expires = self._day("duos")
        self.table.add(self._user(user), key, -count, self.daily + count, expires)
        if request and REQUEST.match(request):
            self.table.delete(self._user(user), f"request#{request}")

    def _visitor_key(self) -> bytes:
        riot_key = getattr(self.riot, "key", None)
        secret = riot_key() if callable(riot_key) else ""
        if not secret:
            return LOCAL_VISITOR_KEY
        return hmac.new(secret.encode("utf-8"), VISITOR_LABEL, hashlib.sha256).digest()

    def visitor(self, address: str) -> str:
        return f"{VISITOR}#" + hmac.new(self._visitor_key(), f"{self.day()}|{address}".encode("utf-8"), hashlib.sha256).hexdigest()

    def _visitor_full(self, visitor: str) -> QuotaExceeded:
        key, _ = self._day("duos")
        used = int((self.table.get(visitor, key) or {}).get("used", 0))
        return QuotaExceeded(f"You have {max(0, self.visitor_duos - used)} of {self.visitor_duos} duo checks left today. Sign in for more.")

    def visitor_spend(self, visitor: str, count: int, request: str | None = None) -> bool:
        if count > self.visitor_duos:
            raise QuotaExceeded(f"You can check {self.visitor_duos} duos a day. Sign in for more.")
        key, expires = self._day("duos")
        if request and REQUEST.match(request):
            outcome = self.table.charge(visitor, key, f"request#{request}", count, self.visitor_duos, expires)
            if outcome == DUPLICATE:
                return False
            if outcome == FULL:
                raise self._visitor_full(visitor)
        elif self.table.add(visitor, key, count, self.visitor_duos, expires) is None:
            raise self._visitor_full(visitor)
        if self.table.add(ANONYMOUS, key, count, self.anonymous_duos, expires) is None:
            self._give_back_visitor(visitor, count, request)
            raise QuotaExceeded(EVERYONE_FULL)
        return True

    def _give_back_visitor(self, visitor: str, count: int, request: str | None) -> None:
        key, expires = self._day("duos")
        if count:
            self.table.add(visitor, key, -count, self.visitor_duos + count, expires)
        if request and REQUEST.match(request):
            self.table.delete(visitor, f"request#{request}")

    def visitor_refund(self, visitor: str, count: int, request: str | None = None) -> None:
        self._give_back_visitor(visitor, count, request)
        if count:
            key, expires = self._day("duos")
            self.table.add(ANONYMOUS, key, -count, self.anonymous_duos + count, expires)
