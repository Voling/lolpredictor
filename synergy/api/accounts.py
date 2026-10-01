import hashlib
import random
import re
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
CHECK_SECONDS = 10
RIOT_BUDGET = "riot#budget"
RIOT_PER_SECOND = 10
CONFLICT_TRIES = 3
CHARGED, DUPLICATE, FULL = "charged", "duplicate", "full"


class AuthError(Exception):
    pass


class QuotaExceeded(Exception):
    pass


class LinkError(Exception):
    pass


class Busy(Exception):
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
            raise AuthError("Sign in to check duos.")
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
        return int(value)
    if kind == "BOOL":
        return bool(value)
    return value


def _item(values: dict) -> dict:
    out = {}
    for name, value in values.items():
        if isinstance(value, bool):
            out[name] = {"BOOL": value}
        elif isinstance(value, int):
            out[name] = {"N": str(value)}
        else:
            out[name] = {"S": str(value)}
    return out


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


class RiotAccounts:
    def __init__(self, key, platform: str, region: str, client: httpx.Client | None = None):
        self.key, self.client = key, client or httpx.Client(timeout=TIMEOUT)
        self.account_host = f"https://{region}.api.riotgames.com"
        self.platform_host = f"https://{platform}.api.riotgames.com"

    def _get(self, url: str) -> dict:
        try:
            response = self.client.get(url, headers={"X-Riot-Token": self.key()})
        except httpx.HTTPError as exc:
            raise LinkError("We couldn't reach Riot. Try again later.") from exc
        if response.status_code == 404:
            raise LinkError("Riot doesn't know that Riot ID. Check the name and tag.")
        if response.status_code == 429:
            raise LinkError("Riot is busy right now. Try again in a few minutes.")
        if response.status_code != 200:
            raise LinkError("We couldn't reach Riot. Try again later.")
        return response.json()

    def account(self, riot_id: str) -> dict:
        name, _, tag = riot_id.partition("#")
        return self._get(f"{self.account_host}/riot/account/v1/accounts/by-riot-id/{quote(name, safe='')}/{quote(tag, safe='')}")

    def icon(self, puuid: str) -> int:
        return int(self._get(f"{self.platform_host}/lol/summoner/v4/summoners/by-puuid/{quote(puuid, safe='')}")["profileIconId"])


def ssm_key(name: str):
    found = {}

    def key() -> str:
        if "value" not in found:
            import boto3

            found["value"] = boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
        return found["value"]

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
    def __init__(self, table, riot, daily: int, attempts: int, riot_budget: int = 30, riot_per_second: int = RIOT_PER_SECOND, clock=time.time, shuffle=_shuffled):
        self.table, self.riot, self.daily, self.attempts = table, riot, daily, attempts
        self.riot_budget, self.riot_per_second = riot_budget, riot_per_second
        self.clock, self.shuffle = clock, shuffle

    def _user(self, user: str) -> str:
        return f"user#{user}"

    def day(self) -> str:
        return f"{datetime.fromtimestamp(self.clock(), timezone.utc):%Y-%m-%d}"

    def _day(self, kind: str) -> tuple[str, int]:
        return f"{kind}#{self.day()}", int(self.clock()) + 2 * DAY

    def _riot_calls(self, calls: int) -> bool:
        now = int(self.clock())
        minute = f"minute#{now // 60}"
        if self.table.add(RIOT_BUDGET, minute, calls, self.riot_budget, now + DAY) is None:
            return False
        if self.table.add(RIOT_BUDGET, f"second#{now}", calls, self.riot_per_second, now + DAY) is None:
            self.table.add(RIOT_BUDGET, minute, -calls, self.riot_budget + calls, now + DAY)
            return False
        return True

    def _attempt(self, user: str, riot_calls: int) -> None:
        key, expires = self._day("links")
        if self.table.add(self._user(user), key, 1, self.attempts, expires) is None:
            raise QuotaExceeded(f"You can try linking {self.attempts} times a day. Try again tomorrow.")
        if not self._riot_calls(riot_calls):
            self.table.add(self._user(user), key, -1, self.attempts + 1, expires)
            raise LinkError("Riot is busy right now. Try again in a few minutes.")

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
        self._attempt(user, 2)
        account = self.riot.account(riot_id)
        puuid = account["puuid"]
        current = self.riot.icon(puuid)
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
        if self.table.add(RIOT_BUDGET, f"checks#{now // 60}", 1, self.riot_budget // 2, now + DAY) is None or not self._riot_calls(1):
            return self.status(user)
        if self.riot.icon(pending["puuid"]) != pending["icon"]:
            return self.status(user)
        if not self.table.claim(f"puuid#{pending['puuid']}", "owner", user):
            raise LinkError("That Riot account is already linked to another lolpredictor account.")
        if link.get("verified") and link.get("puuid") != pending["puuid"]:
            self.table.release(f"puuid#{link['puuid']}", "owner", user)
        self.table.release(f"puuid#{pending['puuid']}", f"pending#{pending['icon']}", user)
        self.table.put(self._user(user), LINK, {"riot_id": pending["riot_id"], "puuid": pending["puuid"], "verified": True})
        return self.status(user)

    def linked_puuid(self, user: str) -> str:
        link = self.link(user)
        if not link.get("verified"):
            raise LinkError("Link your Riot account on the account page first.")
        return link["puuid"]

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
