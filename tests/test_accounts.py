import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from synergy.api import server
from synergy.api.accounts import (
    Accounts,
    AuthError,
    Busy,
    DynamoTable,
    LinkError,
    QuotaExceeded,
    SigningKeys,
    TokenVerifier,
    request_key,
    riot_id_of,
)

ISSUER = "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_test"
DAY = 86_400
START = 1_790_000_000.0


class MemoryTable:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found is not None else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)

    def delete(self, pk, sk):
        self.items.pop((pk, sk), None)

    def claim(self, pk, sk, owner):
        found = self.items.get((pk, sk))
        if found is not None and found["owner"] != owner:
            return False
        self.items[(pk, sk)] = {"owner": owner}
        return True

    def reserve(self, pk, sk, owner, now, expires):
        found = self.items.get((pk, sk))
        if found is not None and found["owner"] != owner and found["expires"] >= now:
            return False
        self.items[(pk, sk)] = {"owner": owner, "expires": expires}
        return True

    def release(self, pk, sk, owner):
        if self.items.get((pk, sk), {}).get("owner") == owner:
            del self.items[(pk, sk)]

    def add(self, pk, sk, amount, limit, expires):
        found = self.items.get((pk, sk))
        if found is not None and found["used"] > limit - amount:
            return None
        used = (found or {"used": 0})["used"] + amount
        self.items[(pk, sk)] = {"used": used, "expires": expires}
        return used

    def charge(self, pk, counter, request, amount, limit, expires):
        if (pk, request) in self.items:
            return "duplicate"
        found = self.items.get((pk, counter))
        if found is not None and found["used"] > limit - amount:
            return "full"
        self.items[(pk, request)] = {"expires": expires}
        self.add(pk, counter, amount, limit, expires)
        return "charged"


class FakeRiot:
    def __init__(self):
        self.icons = {"puuid-a": 7, "puuid-b": 3}
        self.calls = 0

    def account(self, riot_id):
        self.calls += 1
        name, _, tag = riot_id.partition("#")
        if name == "ghost":
            raise LinkError("Riot doesn't know that Riot ID. Check the name and tag.")
        return {"puuid": f"puuid-{name}", "gameName": name, "tagLine": tag.upper()}

    def icon(self, puuid):
        self.calls += 1
        return self.icons[puuid]


def _accounts(clock=lambda: START, attempts=3, riot_budget=1000, riot_per_second=1000):
    return Accounts(MemoryTable(), FakeRiot(), daily=20, attempts=attempts, riot_budget=riot_budget, riot_per_second=riot_per_second, clock=clock, shuffle=lambda items: list(items))


def _token(key, kid="key-1", **claims):
    body = {"iss": ISSUER, "sub": "user-1", "token_use": "access", "exp": int(time.time()) + 600, **claims}
    return jwt.encode(body, key, algorithm="RS256", headers={"kid": kid})


def _link(accounts, user, riot_id):
    started = accounts.start_link(user, riot_id)
    accounts.riot.icons[f"puuid-{riot_id.partition('#')[0]}"] = started["pending"]["icon"]
    return accounts.verify_link(user)


@pytest.fixture(scope="module")
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def test_an_access_token_from_the_pool_names_its_user(keys):
    # given
    private, public = keys
    verifier = TokenVerifier(ISSUER, lambda token: public)

    # when
    user = verifier.subject(_token(private))

    # then
    assert user == "user-1"


@pytest.mark.parametrize(
    "claims",
    [{"token_use": "id"}, {"iss": "https://cognito-idp.us-west-2.amazonaws.com/other"}, {"exp": int(time.time()) - 60}],
)
def test_id_tokens_other_pools_and_expired_tokens_are_refused(keys, claims):
    # given
    private, public = keys
    verifier = TokenVerifier(ISSUER, lambda token: public)

    # when
    with pytest.raises(AuthError) as refused:
        verifier.subject(_token(private, **claims))

    # then
    assert "Sign in" in str(refused.value)


def test_a_missing_or_forged_token_is_refused(keys):
    # given
    _, public = keys
    forger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = TokenVerifier(ISSUER, lambda token: public)

    # when
    with pytest.raises(AuthError):
        verifier.subject(None)
    with pytest.raises(AuthError) as forged:
        verifier.subject(_token(forger))

    # then
    assert "Sign in again" in str(forged.value)


def test_tokens_with_unknown_key_ids_fetch_the_pool_keys_at_most_once_per_refresh(keys):
    # given
    private, public = keys
    fetched = []
    published = {"keys": [{**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public)), "kid": "key-1"}]}
    now = {"t": 0.0}
    verifier = TokenVerifier(ISSUER, SigningKeys(lambda: fetched.append(1) or published, clock=lambda: now["t"]))

    # when
    first = verifier.subject(_token(private))
    for index in range(50):
        with pytest.raises(AuthError):
            verifier.subject(_token(private, kid=f"random-{index}"))
    before_refresh = len(fetched)
    now["t"] += 601.0
    with pytest.raises(AuthError):
        verifier.subject(_token(private, kid="random-late"))

    # then
    assert first == "user-1" and before_refresh == 1 and len(fetched) == 2


def test_a_failed_key_fetch_is_retried_within_seconds_not_minutes(keys):
    # given
    private, public = keys
    published = {"keys": [{**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public)), "kid": "key-1"}]}
    answers = [httpx.ConnectTimeout("slow"), {"keys": []}, published]
    now = {"t": 0.0}

    def fetch():
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    verifier = TokenVerifier(ISSUER, SigningKeys(fetch, clock=lambda: now["t"]))

    # when
    with pytest.raises(AuthError) as outage:
        verifier.subject(_token(private))
    now["t"] += 11.0
    with pytest.raises(AuthError):
        verifier.subject(_token(private))
    now["t"] += 11.0
    user = verifier.subject(_token(private))

    # then
    assert "couldn't check your sign in" in str(outage.value) and user == "user-1" and answers == []


def test_a_riot_id_needs_a_name_and_a_tag():
    # given
    texts = ["  Faker  #  KR1 ", "Faker", "#KR1", "x" * 30 + "#" + "y" * 20]

    # when
    first = riot_id_of(texts[0])

    # then
    assert first == "Faker#KR1"
    for text in texts[1:]:
        with pytest.raises(LinkError):
            riot_id_of(text)


def test_linking_asks_for_a_new_icon_and_only_verifies_once_it_is_set():
    # given
    accounts = _accounts()

    # when
    started = accounts.start_link("user-1", "a#na1")
    with pytest.raises(LinkError) as early:
        accounts.verify_link("user-1")
    accounts.riot.icons["puuid-a"] = started["pending"]["icon"]
    done = accounts.verify_link("user-1")

    # then
    assert started["pending"] == {"riot_id": "a#NA1", "icon": 0} and started["verified"] is False
    assert "icon 0" in str(early.value)
    assert done["verified"] is True and done["riot_id"] == "a#NA1" and done["pending"] is None
    assert accounts.linked_puuid("user-1") == "puuid-a"


def test_pending_links_on_one_riot_account_never_share_an_icon_so_the_owner_cannot_be_raced():
    # given
    accounts = _accounts(attempts=10)
    attackers = [accounts.start_link(f"attacker-{index}", "a#na1")["pending"]["icon"] for index in range(5)]

    # when
    owner = accounts.start_link("owner", "a#na1")["pending"]["icon"]
    accounts.riot.icons["puuid-a"] = owner
    for index in range(5):
        with pytest.raises(LinkError):
            accounts.verify_link(f"attacker-{index}")
    done = accounts.verify_link("owner")

    # then
    assert len(set(attackers + [owner])) == 6 and done["verified"] is True
    assert accounts.linked_puuid("owner") == "puuid-a"


def test_a_pending_link_expires_after_15_minutes():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])
    started = accounts.start_link("user-1", "a#na1")
    accounts.riot.icons["puuid-a"] = started["pending"]["icon"]

    # when
    now["t"] += 16 * 60
    with pytest.raises(LinkError) as expired:
        accounts.verify_link("user-1")

    # then
    assert "expired" in str(expired.value) and accounts.status("user-1")["pending"] is None


def test_starting_a_new_link_keeps_the_verified_one_until_the_new_one_is_proven():
    # given
    accounts = _accounts(attempts=10)
    _link(accounts, "user-1", "a#na1")

    # when
    accounts.start_link("user-1", "b#na1")
    during = accounts.linked_puuid("user-1")
    accounts.riot.icons["puuid-b"] = accounts.status("user-1")["pending"]["icon"]
    accounts.verify_link("user-1")

    # then
    assert during == "puuid-a" and accounts.linked_puuid("user-1") == "puuid-b"
    assert accounts.table.get("puuid#puuid-a", "owner") is None


def test_a_riot_account_links_to_only_one_lolpredictor_account():
    # given
    accounts = _accounts()
    _link(accounts, "user-1", "a#na1")

    # when
    second = accounts.start_link("user-2", "a#na1")
    accounts.riot.icons["puuid-a"] = second["pending"]["icon"]
    with pytest.raises(LinkError) as taken:
        accounts.verify_link("user-2")

    # then
    assert "already linked" in str(taken.value)
    with pytest.raises(LinkError):
        accounts.linked_puuid("user-2")


def test_starting_and_verifying_links_share_one_daily_allowance():
    # given
    accounts = _accounts()
    accounts.start_link("user-1", "b#na1")

    # when
    for _ in range(2):
        with pytest.raises(LinkError):
            accounts.verify_link("user-1")
    with pytest.raises(QuotaExceeded) as capped:
        accounts.verify_link("user-1")

    # then
    assert "3 times a day" in str(capped.value) and accounts.riot.calls == 4


def test_riot_calls_stop_at_the_shared_budget_and_the_refused_attempt_is_given_back():
    # given
    accounts = _accounts(attempts=10, riot_budget=4)
    accounts.start_link("user-1", "a#na1")
    accounts.start_link("user-2", "b#na1")

    # when
    with pytest.raises(LinkError) as busy:
        accounts.start_link("user-3", "a#na1")

    # then
    assert "Riot is busy" in str(busy.value) and accounts.riot.calls == 4
    assert accounts.table.get("user#user-3", f"links#{accounts.day()}")["used"] == 0


def test_riot_calls_are_also_capped_within_each_second():
    # given
    accounts = _accounts(attempts=10, riot_budget=100, riot_per_second=10)
    for index in range(5):
        accounts.start_link(f"user-{index}", "a#na1")

    # when
    with pytest.raises(LinkError) as busy:
        accounts.start_link("user-5", "a#na1")

    # then
    assert "Riot is busy" in str(busy.value) and accounts.riot.calls == 10


def test_duo_checks_stop_at_the_daily_allowance_refund_on_failure_and_reset_the_next_day():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])

    # when
    left, charged = accounts.spend("user-1", 15)
    with pytest.raises(QuotaExceeded) as over:
        accounts.spend("user-1", 6)
    accounts.refund("user-1", 5)
    after_refund = accounts.status("user-1")["remaining"]
    now["t"] += DAY
    tomorrow = accounts.status("user-1")["remaining"]

    # then
    assert left == 5 and charged and "5 of 20" in str(over.value)
    assert after_refund == 10 and tomorrow == 20
    with pytest.raises(QuotaExceeded):
        accounts.spend("user-1", 21)


def test_the_same_query_on_the_same_day_is_charged_once_and_a_new_day_charges_again():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])
    first_key = request_key("pair", accounts.day(), "Friend#NA1", None, None)

    # when
    first = accounts.spend("user-1", 1, first_key)
    again = accounts.spend("user-1", 1, request_key("pair", accounts.day(), "  friend#na1 ", None, None))
    other = accounts.spend("user-1", 1, request_key("pair", accounts.day(), "other#na1", None, None))
    now["t"] += DAY
    tomorrow = accounts.spend("user-1", 1, request_key("pair", accounts.day(), "Friend#NA1", None, None))

    # then
    assert first == (19, True) and again == (19, False) and other == (18, True) and tomorrow == (19, True)


def test_a_refused_or_refunded_query_never_buys_a_free_check():
    # given
    accounts = _accounts()
    key = request_key("pair", accounts.day(), "friend#na1", None, None)
    accounts.spend("user-1", 20)

    # when
    with pytest.raises(QuotaExceeded):
        accounts.spend("user-1", 1, key)
    with pytest.raises(QuotaExceeded):
        accounts.spend("user-1", 1, key)
    fresh = _accounts()
    fresh.spend("user-2", 1, key)
    fresh.refund("user-2", 1, key)
    again = fresh.spend("user-2", 1, key)

    # then
    assert again == (19, True)


class _Service:
    def __init__(self):
        import pandas as pd

        self.ready = True
        self.profiles = pd.DataFrame(index=["puuid-a"])


def _metered_setup(monkeypatch):
    accounts = _accounts(attempts=10)
    _link(accounts, "user-1", "a#na1")
    monkeypatch.setattr(server, "get_verifier", lambda: type("V", (), {"subject": staticmethod(lambda token: "user-1")})())
    monkeypatch.setattr(server, "get_accounts", lambda: accounts)
    monkeypatch.setattr(server, "get_service", lambda: _Service())
    return accounts


def test_a_metered_check_that_fails_is_refunded_and_can_be_retried_at_full_price(monkeypatch):
    # given
    accounts = _metered_setup(monkeypatch)

    def broken(me, service):
        raise ValueError("boom")

    # when
    with pytest.raises(ValueError):
        server._metered("token", ("pair", "friend#na1", None, None), 1, broken)
    after_failure = accounts.status("user-1")["remaining"]
    found = server._metered("token", ("pair", "friend#na1", None, None), 1, lambda me, service: {"me": me})

    # then
    assert after_failure == 20 and found == {"me": "puuid-a", "remaining": 19}


def test_friends_that_could_not_be_scored_are_given_back(monkeypatch):
    # given
    accounts = _metered_setup(monkeypatch)
    rows = {"friends": [{"score": 60.0}, {"score": None, "note": "Not in our data yet."}, {"note": "typo"}]}

    # when
    found = server._metered("token", ("friends", None, "a", "b", "c"), 3, lambda me, service: rows, lambda answer: sum(1 for row in answer["friends"] if row.get("score") is None))

    # then
    assert found["remaining"] == 19 and accounts.status("user-1")["remaining"] == 19


class _Failed(Exception):
    pass


class _Conflict(Exception):
    pass


class _Cancelled(Exception):
    def __init__(self, codes):
        super().__init__("cancelled")
        self.response = {"CancellationReasons": [{"Code": code} for code in codes]}


class _Recorder:
    class exceptions:
        ConditionalCheckFailedException = _Failed
        TransactionCanceledException = _Cancelled
        TransactionConflictException = _Conflict

    def __init__(self, fail=None, conflicts=0):
        self.calls, self.fail, self.conflicts = [], fail, conflicts

    def update_item(self, **request):
        self.calls.append(request)
        if self.conflicts:
            self.conflicts -= 1
            raise _Conflict()
        if self.fail:
            raise _Failed()
        return {"Attributes": {"used": {"N": "4"}}}

    def transact_write_items(self, **request):
        self.calls.append(request)
        if self.conflicts:
            self.conflicts -= 1
            raise _Cancelled(["None", "TransactionConflict"])
        if self.fail:
            raise _Cancelled(self.fail)


def _table(recorder):
    return DynamoTable("accounts", recorder, pause=lambda seconds: None)


def test_the_dynamo_counter_adds_atomically_under_the_daily_room():
    # given
    recorder = _Recorder()

    # when
    used = _table(recorder).add("user#1", "duos#2026-09-27", 2, 20, 99)
    refused = _table(_Recorder(fail=True)).add("user#1", "duos#2026-09-27", 2, 20, 99)

    # then
    request = recorder.calls[0]
    assert used == 4 and refused is None
    assert request["UpdateExpression"] == "ADD used :amount SET expires = :expires"
    assert request["ConditionExpression"] == "attribute_not_exists(used) OR used <= :room"
    assert request["ExpressionAttributeValues"][":room"] == {"N": "18"}


def test_a_charge_writes_the_request_and_the_count_in_one_transaction_and_names_why_it_was_refused():
    # given
    recorder = _Recorder()

    # when
    charged = _table(recorder).charge("user#1", "duos#d", "request#r", 2, 20, 99)
    duplicate = _table(_Recorder(fail=["ConditionalCheckFailed", "None"])).charge("user#1", "duos#d", "request#r", 2, 20, 99)
    full = _table(_Recorder(fail=["None", "ConditionalCheckFailed"])).charge("user#1", "duos#d", "request#r", 2, 20, 99)

    # then
    items = recorder.calls[0]["TransactItems"]
    assert (charged, duplicate, full) == ("charged", "duplicate", "full")
    assert items[0]["Put"]["ConditionExpression"] == "attribute_not_exists(pk)"
    assert items[0]["Put"]["Item"]["sk"] == {"S": "request#r"}
    assert items[1]["Update"]["Key"]["sk"] == {"S": "duos#d"}


def test_write_conflicts_are_retried_then_reported_as_busy():
    # given
    brief, stuck = _Recorder(conflicts=2), _Recorder(conflicts=5)

    # when
    charged = _table(brief).charge("user#1", "duos#d", "request#r", 1, 20, 99)
    added = _table(_Recorder(conflicts=1)).add("user#1", "duos#d", 1, 20, 99)
    with pytest.raises(Busy):
        _table(stuck).charge("user#1", "duos#d", "request#r", 1, 20, 99)

    # then
    assert charged == "charged" and len(brief.calls) == 3 and added == 4
