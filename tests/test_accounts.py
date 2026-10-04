import hashlib
import hmac
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from synergy.api import server
from synergy.api.accounts import (
    Accounts,
    AuthError,
    CHECK_SECONDS,
    Busy,
    DynamoTable,
    LinkError,
    QuotaExceeded,
    RiotBusy,
    SigningKeys,
    TokenVerifier,
    request_key,
    riot_id_of,
)
from synergy.api.evaluate import Evaluations
from synergy.api.history import History
from synergy.ml.score import UNKNOWN_NOTE, UnknownPlayer

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

    def latest(self, pk, limit):
        keys = sorted((key for key in self.items if key[0] == pk), key=lambda key: key[1], reverse=True)[:limit]
        return [{"pk": key[0], "sk": key[1], **self.items[key]} for key in keys]

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


def test_linking_asks_for_a_new_icon_and_only_verifies_once_riot_shows_it():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])

    # when
    started = accounts.start_link("user-1", "a#na1")
    early = accounts.verify_link("user-1")
    accounts.riot.icons["puuid-a"] = started["pending"]["icon"]
    now["t"] += CHECK_SECONDS
    done = accounts.verify_link("user-1")

    # then
    assert started["pending"] == {"riot_id": "a#NA1", "icon": 0} and started["verified"] is False
    assert early["verified"] is False and early["pending"] == started["pending"]
    assert done["verified"] is True and done["riot_id"] == "a#NA1" and done["pending"] is None
    assert accounts.linked_puuid("user-1") == "puuid-a"


def test_pending_links_on_one_riot_account_never_share_an_icon_so_the_owner_cannot_be_raced():
    # given
    accounts = _accounts(attempts=10)
    attackers = [accounts.start_link(f"attacker-{index}", "a#na1")["pending"]["icon"] for index in range(5)]

    # when
    owner = accounts.start_link("owner", "a#na1")["pending"]["icon"]
    accounts.riot.icons["puuid-a"] = owner
    refused = [accounts.verify_link(f"attacker-{index}") for index in range(5)]
    done = accounts.verify_link("owner")

    # then
    assert len(set(attackers + [owner])) == 6 and done["verified"] is True
    assert not any(found["verified"] for found in refused)
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
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"], attempts=10)
    _link(accounts, "user-1", "a#na1")

    # when
    accounts.start_link("user-1", "b#na1")
    during = accounts.linked_puuid("user-1")
    accounts.riot.icons["puuid-b"] = accounts.status("user-1")["pending"]["icon"]
    now["t"] += CHECK_SECONDS
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
    assert "already linked" in str(taken.value) and accounts.linked_puuid("user-2") is None


def test_icon_checks_never_use_the_daily_allowance_and_reach_riot_at_most_every_10_seconds():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])
    accounts.start_link("user-1", "b#na1")

    # when
    for _ in range(3):
        accounts.verify_link("user-1")
    now["t"] += CHECK_SECONDS
    waiting = accounts.verify_link("user-1")

    # then
    assert accounts.riot.calls == 4 and waiting["verified"] is False and waiting["pending"] is not None
    assert accounts.table.get("user#user-1", f"links#{accounts.day()}")["used"] == 1


def test_entering_the_same_riot_id_again_keeps_its_icon_and_costs_nothing():
    # given
    accounts = _accounts()
    first = accounts.start_link("user-1", "a#na1")

    # when
    again = accounts.start_link("user-1", " A#NA1 ")

    # then
    assert again["pending"] == first["pending"] and accounts.riot.calls == 2
    assert accounts.table.get("user#user-1", f"links#{accounts.day()}")["used"] == 1


def test_the_daily_allowance_caps_new_links():
    # given
    accounts = _accounts()
    for name in ("a", "b", "a"):
        accounts.start_link("user-1", f"{name}#na1")

    # when
    with pytest.raises(QuotaExceeded) as capped:
        accounts.start_link("user-1", "b#na1")

    # then
    assert "3 times a day" in str(capped.value) and accounts.riot.calls == 6


def test_an_icon_check_waits_instead_of_failing_when_riot_is_busy():
    # given
    accounts = _accounts(riot_budget=2)
    started = accounts.start_link("user-1", "a#na1")
    accounts.riot.icons["puuid-a"] = started["pending"]["icon"]

    # when
    waiting = accounts.verify_link("user-1")

    # then
    assert waiting["verified"] is False and waiting["pending"] == started["pending"] and accounts.riot.calls == 2


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


def test_icon_checks_use_at_most_half_the_riot_budget_so_new_links_still_start():
    # given
    accounts = _accounts(attempts=10, riot_budget=8)
    for index in range(2):
        accounts.start_link(f"user-{index}", "a#na1")

    # when
    waiting = [accounts.verify_link(f"user-{index}") for index in range(2)]
    started = accounts.start_link("user-9", "b#na1")

    # then
    assert accounts.riot.calls == 8 and all(found["pending"] for found in waiting) and started["pending"] is not None


def test_recent_duos_come_back_newest_first_capped_at_five_and_without_names():
    # given
    now = {"t": START}
    accounts = _accounts(clock=lambda: now["t"])
    for index in range(7):
        now["t"] += 1
        accounts.remember([{"left_champion": "Ahri", "left_tier": "DIAMOND", "left_division": "II", "left_position": "MIDDLE", "right_champion": "LeeSin", "right_tier": "MASTER", "right_division": None, "right_position": "JUNGLE", "score": 50.0 + index, "gold": 12.5, "minute": 20}])

    # when
    found = accounts.recent()

    # then
    assert [duo["score"] for duo in found] == [56.0, 55.0, 54.0, 53.0, 52.0]
    assert found[0]["at"] == int(START) + 7 and "riot_id" not in found[0] and "right_division" not in found[0]
    assert set(found[0]) == {"left_champion", "left_tier", "left_division", "left_position", "right_champion", "right_tier", "right_position", "score", "gold", "minute", "at"}


def test_dynamo_items_round_trip_floats_and_drop_missing_values():
    # given
    from synergy.api.accounts import _item, _value

    stored = _item({"score": 57.9, "games": 10, "ok": True, "name": "x", "gone": None})

    # when
    back = {name: _value(value) for name, value in stored.items()}

    # then
    assert stored["score"] == {"N": "57.9"} and "gone" not in stored
    assert back == {"score": 57.9, "games": 10, "ok": True, "name": "x"}


def test_the_dynamo_table_reads_the_latest_rows_in_reverse_key_order():
    # given
    class _Client:
        exceptions = _Recorder.exceptions

        def query(self, **request):
            self.request = request
            return {"Items": [{"pk": {"S": "recent"}, "sk": {"S": "2#0"}, "score": {"N": "61.5"}}, {"pk": {"S": "recent"}, "sk": {"S": "1#0"}, "score": {"N": "48"}}]}

    client = _Client()
    table = DynamoTable("accounts", client=client)

    # when
    found = table.latest("recent", 5)

    # then
    assert [row["score"] for row in found] == [61.5, 48]
    assert client.request["ScanIndexForward"] is False and client.request["Limit"] == 5 and client.request["ExpressionAttributeValues"] == {":pk": {"S": "recent"}}


def test_the_dynamo_table_can_read_only_some_fields_of_the_latest_rows():
    # given
    class _Client:
        exceptions = _Recorder.exceptions

        def query(self, **request):
            self.request = request
            return {"Items": [{"sk": {"S": "2026-10-01#abc"}, "at": {"N": "1790000000"}, "kind": {"S": "pair"}}]}

    client = _Client()

    # when
    found = DynamoTable("accounts", client=client).latest("checks#u", 50, ("at", "kind", "summary"))

    # then
    assert found == [{"sk": "2026-10-01#abc", "at": 1790000000, "kind": "pair"}]
    assert client.request["ProjectionExpression"] == "sk, #f0, #f1, #f2"
    assert client.request["ExpressionAttributeNames"] == {"#f0": "at", "#f1": "kind", "#f2": "summary"}


def _visitors(clock=lambda: START, duos=20, everyone=3000, riot=None, riot_budget=1000, visitor_riot=20):
    return Accounts(
        MemoryTable(),
        riot or FakeRiot(),
        daily=20,
        attempts=3,
        riot_budget=riot_budget,
        riot_per_second=1000,
        clock=clock,
        visitor_duos=duos,
        anonymous_duos=everyone,
        visitor_riot_budget=visitor_riot,
    )


def test_a_visitor_stops_at_the_daily_allowance_while_other_visitors_still_check():
    # given
    accounts = _visitors(duos=2)
    one, two = accounts.visitor("203.0.113.7"), accounts.visitor("198.51.100.4")

    # when
    first = accounts.visitor_spend(one, 1, request_key("pair", accounts.day(), "a#na1", "b#na1"))
    repeated = accounts.visitor_spend(one, 1, request_key("pair", accounts.day(), "a#na1", "b#na1"))
    second = accounts.visitor_spend(one, 1, request_key("pair", accounts.day(), "a#na1", "c#na1"))
    with pytest.raises(QuotaExceeded) as over:
        accounts.visitor_spend(one, 1, request_key("pair", accounts.day(), "a#na1", "d#na1"))
    other = accounts.visitor_spend(two, 1, request_key("pair", accounts.day(), "a#na1", "d#na1"))

    # then
    assert (first, repeated, second, other) == (True, False, True, True)
    assert "0 of 2 duo checks left today. Sign in for more." in str(over.value)


def test_checks_without_an_account_stop_for_everyone_at_the_shared_cap_and_the_refused_visitor_keeps_their_allowance():
    # given
    accounts = _visitors(everyone=2)
    first, second, third = (accounts.visitor(address) for address in ("203.0.113.7", "198.51.100.4", "192.0.2.1"))
    key = request_key("pair", accounts.day(), "a#na1", "b#na1")
    accounts.visitor_spend(first, 1, key)
    accounts.visitor_spend(second, 1, key)

    # when
    with pytest.raises(QuotaExceeded) as full:
        accounts.visitor_spend(third, 1, key)
    accounts.visitor_refund(first, 1, key)
    retried = accounts.visitor_spend(third, 1, key)

    # then
    assert "full for today" in str(full.value) and retried is True
    assert accounts.table.get(third, f"duos#{accounts.day()}")["used"] == 1


def test_a_visitor_allowance_starts_again_the_next_day():
    # given
    now = {"t": START}
    accounts = _visitors(clock=lambda: now["t"], duos=1, everyone=1)
    today = accounts.visitor("203.0.113.7")
    accounts.visitor_spend(today, 1)
    with pytest.raises(QuotaExceeded):
        accounts.visitor_spend(today, 1)

    # when
    now["t"] += DAY
    tomorrow = accounts.visitor("203.0.113.7")
    charged = accounts.visitor_spend(tomorrow, 1)

    # then
    assert charged is True and tomorrow != today


class _KeyedRiot(FakeRiot):
    def __init__(self, secret):
        super().__init__()
        self.secret = secret

    def key(self):
        return self.secret


def test_a_visitor_is_stored_under_a_key_that_changes_with_the_secret_and_the_day_and_never_shows_either():
    # given
    now = {"t": START}
    one = _visitors(clock=lambda: now["t"], riot=_KeyedRiot("RGAPI-one"))
    two = _visitors(clock=lambda: now["t"], riot=_KeyedRiot("RGAPI-two"))
    local = _visitors(clock=lambda: now["t"])
    derived = hmac.new(b"RGAPI-one", b"lolpredictor visitor v1", hashlib.sha256).digest()
    expected = "visitor#" + hmac.new(derived, f"{one.day()}|203.0.113.7".encode("utf-8"), hashlib.sha256).hexdigest()

    # when
    visitor = one.visitor("203.0.113.7")
    one.visitor_spend(visitor, 1, request_key("pair", one.day(), "a#na1", "b#na1"))
    others = (two.visitor("203.0.113.7"), local.visitor("203.0.113.7"))
    now["t"] += DAY
    tomorrow = one.visitor("203.0.113.7")

    # then
    stored = " ".join(f"{key} {value}" for key, value in one.table.items.items())
    assert visitor == expected and len({visitor, *others, tomorrow}) == 4 and visitor in stored
    assert not any(secret in stored for secret in ("RGAPI-one", derived.hex(), "203.0.113.7"))


def test_visitors_together_use_at_most_their_slice_of_the_riot_budget_each_minute():
    # given
    now = {"t": START}
    accounts = _visitors(clock=lambda: now["t"], visitor_riot=2)

    # when
    allowed = [accounts.visitor_riot_calls(1), accounts.visitor_riot_calls(1)]
    with pytest.raises(RiotBusy) as busy:
        accounts.visitor_riot_calls(1)
    members = accounts.riot_calls(1)
    now["t"] += 60
    next_minute = accounts.visitor_riot_calls(1)

    # then
    assert allowed == [True, True] and members is True and next_minute is True
    assert str(busy.value) == "Busy, try again in a minute."


def test_a_visitor_riot_call_is_refused_and_given_back_when_the_shared_budget_is_full():
    # given
    accounts = _visitors(riot_budget=1, visitor_riot=5)
    accounts.riot_calls(1)

    # when
    with pytest.raises(RiotBusy):
        accounts.visitor_riot_calls(1)

    # then
    assert accounts.table.get("riot#budget", f"visitors#{int(START) // 60}")["used"] == 0


class _Players:
    def __init__(self, known):
        import pandas as pd

        self.ready = True
        self.known = known
        self.profiles = pd.DataFrame(index=list(known.values()))

    def resolve(self, name):
        if name.lower() not in self.known:
            raise UnknownPlayer(name)
        return {"puuid": self.known[name.lower()]}


class _Queue:
    evaluate_queue = "queue"
    evaluated_store = ""
    accounts_table = ""


def _visitor_setup(monkeypatch, known, made_up=(), **limits):
    accounts = _visitors(**limits)
    asked = []

    def pending(owner, name):
        asked.append((owner, name))
        return {"riot_id": name, "status": "refused" if name in made_up else "requested"}

    monkeypatch.setattr(server, "get_accounts", lambda: accounts)
    monkeypatch.setattr(server, "get_service", lambda: _Players(known))
    monkeypatch.setattr(server, "_pending", pending)
    return accounts, accounts.visitor("203.0.113.7"), asked


def _unknown(name):
    def call(me, service):
        raise UnknownPlayer(name)

    return call


def test_a_visitor_check_that_waits_on_a_new_player_gives_the_check_back(monkeypatch):
    # given
    accounts, visitor, asked = _visitor_setup(monkeypatch, {"me#na1": "puuid-me"})

    # when
    found = server._visited(visitor, "Me#NA1", ("pair", "Me#NA1", "new#na1", None, None), 1, _unknown("new#na1"))

    # then
    assert found == {"score": None, "pending": {"riot_id": "new#na1", "status": "requested"}} and asked == [(visitor, "new#na1")]
    assert accounts.table.get(visitor, f"duos#{accounts.day()}")["used"] == 0 and accounts.table.get("anonymous", f"duos#{accounts.day()}")["used"] == 0


def test_a_visitor_we_have_not_seen_waits_on_their_own_games_and_gets_the_check_back(monkeypatch):
    # given
    accounts, visitor, asked = _visitor_setup(monkeypatch, {})

    # when
    found = server._visited(visitor, " new#na1 ", ("friends", "new#na1", None, "a#na1"), 1, lambda me, service: {"friends": []})

    # then
    assert found["pending"]["riot_id"] == "new#na1" and asked == [(visitor, "new#na1")]
    assert accounts.table.get(visitor, f"duos#{accounts.day()}")["used"] == 0


def test_a_visitor_check_on_a_made_up_name_keeps_the_check_every_time_while_a_real_new_player_gives_it_back(monkeypatch):
    # given
    accounts, visitor, asked = _visitor_setup(monkeypatch, {"me#na1": "puuid-me"}, made_up={"fake#na1", "nobody#na1"})

    # when
    for name in ("fake#na1", "fake#na1", "new#na1"):
        server._visited(visitor, "me#na1", ("pair", "me#na1", name, None, None), 1, _unknown(name))
    server._visited(visitor, "nobody#na1", ("pair", "nobody#na1", "new#na1", None, None), 1, _unknown("new#na1"))

    # then
    assert [name for _, name in asked] == ["fake#na1", "fake#na1", "new#na1", "nobody#na1"]
    assert accounts.table.get(visitor, f"duos#{accounts.day()}")["used"] == 3


def test_a_visitor_ranking_friends_spends_one_check_per_friend_scored_or_made_up(monkeypatch):
    # given
    accounts, visitor, _ = _visitor_setup(monkeypatch, {"me#na1": "puuid-me"}, made_up={"Fake#NA1"})
    monkeypatch.setattr(server, "get_settings", lambda: _Queue())
    rows = [{"riot_id": "A#NA1", "score": 60.0}, {"riot_id": "Fake#NA1", "note": UNKNOWN_NOTE}, {"riot_id": "New#NA1", "note": UNKNOWN_NOTE}, {"riot_id": "C#NA1", "note": "typo"}]

    # when
    found = server._visited(visitor, "me#na1", ("friends", "me#na1", None, "a#na1", "c#na1", "fake#na1", "new#na1"), 4, lambda me, service: {"me": {"riot_id": "Me#NA1"}, "friends": rows}, server._unscored_friends)

    # then
    assert [row.get("pending", {}).get("status") for row in found["friends"]] == [None, "refused", "requested", None]
    assert "remaining" not in found and accounts.table.get(visitor, f"duos#{accounts.day()}")["used"] == 2


def test_a_visitor_check_answers_busy_and_gives_everything_back_once_visitors_used_their_riot_calls(monkeypatch):
    # given
    real = server._pending
    accounts, visitor, _ = _visitor_setup(monkeypatch, {"me#na1": "puuid-me"}, visitor_riot=0)
    evaluations = Evaluations(accounts.table, accounts.riot, accounts.riot_calls, launch=[].append, visitor_budget=accounts.visitor_riot_calls, clock=lambda: START)
    monkeypatch.setattr(server, "_pending", real)
    monkeypatch.setattr(server, "get_settings", lambda: _Queue())
    monkeypatch.setattr(server, "get_evaluations", lambda: evaluations)

    # when
    with pytest.raises(HTTPException) as answered:
        server._handle(lambda: server._visited(visitor, "me#na1", ("pair", "me#na1", "new#na1", None, None), 1, _unknown("new#na1")))

    # then
    assert (answered.value.status_code, answered.value.detail) == (503, "Busy, try again in a minute.") and accounts.riot.calls == 0
    assert accounts.table.get(visitor, f"duos#{accounts.day()}")["used"] == 0 and accounts.table.get(visitor, f"evals#{accounts.day()}")["used"] == 0


def test_an_account_without_a_linked_riot_id_checks_by_name_on_its_own_allowance_and_history(monkeypatch):
    # given
    accounts = _accounts()
    monkeypatch.setattr(server, "get_verifier", lambda: type("V", (), {"subject": staticmethod(lambda token: "user-2")})())
    monkeypatch.setattr(server, "get_accounts", lambda: accounts)
    monkeypatch.setattr(server, "get_service", lambda: _Players({"me#na1": "puuid-me"}))
    monkeypatch.setattr(server, "get_history", lambda: History(accounts.table))

    # when
    found = server._metered("token", ("pair", "friend#na1", None, None), 1, lambda me, service: {"players": [{"riot_id": "Me#NA1"}], "me": me}, name="Me#NA1")

    # then
    assert found["me"] == "puuid-me" and found["remaining"] == 19
    assert [sk for pk, sk in accounts.table.items if pk == "checks#user-2"]


def test_a_linked_account_checks_any_duo_on_its_own_allowance_and_marks_only_duos_it_plays_in(monkeypatch):
    # given
    accounts = _accounts(attempts=10)
    _link(accounts, "user-1", "a#na1")
    monkeypatch.setattr(server, "get_verifier", lambda: type("V", (), {"subject": staticmethod(lambda token: "user-1")})())
    monkeypatch.setattr(server, "get_accounts", lambda: accounts)
    monkeypatch.setattr(server, "get_service", lambda: _Players({"a#na1": "puuid-a", "b#na1": "puuid-b", "x#na1": "puuid-x"}))
    monkeypatch.setattr(server, "get_history", lambda: History(accounts.table))

    def with_player_two(two):
        def call(me, service):
            return {"players": [{"puuid": me, "riot_id": me}, {"puuid": two, "riot_id": two}]}

        return call

    # when
    own = server._metered("token", ("pair", "b#na1", None, None), 1, with_player_two("puuid-b"))
    other = server._metered("token", ("pair", "b#na1", None, None), 1, with_player_two("puuid-b"), name="X#NA1")
    with_me = server._metered("token", ("pair", "a#na1", None, None), 1, with_player_two("puuid-a"), name="x#na1")

    # then
    stored = [json.loads(item["summary"]) for (pk, _), item in accounts.table.items.items() if pk == "checks#user-1"]
    assert sorted((tuple(summary["names"]), summary["mine"]) for summary in stored) == [
        (("puuid-a", "puuid-b"), True),
        (("puuid-x", "puuid-a"), True),
        (("puuid-x", "puuid-b"), False),
    ]
    assert (own["remaining"], other["remaining"], with_me["remaining"]) == (19, 18, 17)


def test_the_visitor_address_is_read_only_from_the_header_the_edge_sets():
    # given
    from starlette.requests import Request

    edge = Request({"type": "http", "headers": [(b"x-viewer-ip", b"203.0.113.7"), (b"x-forwarded-for", b"198.51.100.1"), (b"cloudfront-viewer-address", b"192.0.2.1:443")], "client": ("10.0.0.1", 443)})
    local = Request({"type": "http", "headers": [(b"x-forwarded-for", b"198.51.100.1")], "client": ("127.0.0.1", 50000)})

    # when
    addresses = (server._address(edge), server._address(local))

    # then
    assert addresses == ("203.0.113.7", "127.0.0.1")
