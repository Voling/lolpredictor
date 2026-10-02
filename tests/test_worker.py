import gzip
import json

import pytest

from synergy.api.accounts import LinkError, RateLimited
from synergy.api.evaluate import EVAL, RUNNING
from synergy.worker import GameStore, Progress, fetch_games, requeue, wait_for_budget

START = 1_790_000_000.0


class _Table:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)

    def add(self, pk, sk, amount, limit, expires):
        used = (self.items.get((pk, sk)) or {"used": 0})["used"] + amount
        self.items[(pk, sk)] = {"used": used, "expires": expires}
        return used


class _S3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        class _Body:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

        return {"Body": _Body(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[Key] = Body


class _Riot:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def match_ids(self, puuid, kind, count=100):
        self.calls.append(("ids", kind, count))
        return list(self.ids)

    def match(self, match_id):
        self.calls.append(("match", match_id))
        return {"metadata": {"matchId": match_id}, "info": {"participants": []}}

    def timeline(self, match_id):
        self.calls.append(("timeline", match_id))
        return {"info": {"frames": []}}


class _Accounts:
    def __init__(self, allow=True):
        self.allow, self.asked = allow, []

    def riot_calls(self, calls):
        self.asked.append(calls)
        return self.allow


def test_games_already_in_the_store_are_not_fetched_again_and_new_ones_are_stored_compressed():
    # given
    table, s3 = _Table(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    s3.objects["games/NA1_1.json.gz"] = gzip.compress(json.dumps({"match": {"metadata": {"matchId": "NA1_1"}, "info": {"participants": []}}, "timeline": {"info": {"frames": [1]}}}).encode())
    riot, accounts, seen = _Riot(["NA1_1", "NA1_2"]), _Accounts(), []

    # when
    naps = []
    games = fetch_games(riot, accounts, store, "me", lambda step, done, total: seen.append((step, done, total)), pause=naps.append)

    # then
    assert [match_id for match_id, _, _ in games] == ["NA1_1", "NA1_2"] and games[0][2] == {"info": {"frames": [1]}}
    assert riot.calls == [("ids", "ranked", 50), ("match", "NA1_2"), ("timeline", "NA1_2")] and accounts.asked == [1, 1, 1]
    assert naps == [1.25, 1.25, 1.25]
    assert "games/NA1_2.json.gz" in s3.objects and table.get("evaluated", "bytes")["used"] == len(s3.objects["games/NA1_2.json.gz"])
    assert seen == [("fetching", 1, 2), ("fetching", 2, 2)]


def test_a_full_store_keeps_cached_games_and_stops_pulling_new_ones():
    # given
    table, s3 = _Table(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    riot, accounts = _Riot(["NA1_1", "NA1_2"]), _Accounts()

    # when
    games = fetch_games(riot, accounts, store, "me", lambda *parts: None, store_full=lambda: True, pause=lambda seconds: None)

    # then
    assert games == [] and ("match", "NA1_1") not in riot.calls


def test_a_refused_riot_budget_waits_then_gives_up():
    # given
    now = {"t": START}
    naps = []

    def pause(seconds):
        naps.append(seconds)
        now["t"] += 100

    # when
    with pytest.raises(LinkError) as busy:
        wait_for_budget(_Accounts(allow=False), 2, clock=lambda: now["t"], pause=pause)

    # then
    assert "Riot is busy" in str(busy.value) and len(naps) == 2


def test_progress_maps_each_step_onto_the_overall_percent():
    # given
    table = _Table()
    table.put(EVAL, "me", {"status": "requested", "riot_id": "Me#NA1", "started": int(START)})
    progress = Progress(table, "me", clock=lambda: START)

    # when
    progress.step("fetching", 25, 50)
    middle = table.get(EVAL, "me")
    progress.step("embedding")
    late = table.get(EVAL, "me")

    # then
    assert middle["status"] == RUNNING and middle["step"] == "fetching" and middle["percent"] == 25 and middle["riot_id"] == "Me#NA1"
    assert late["step"] == "embedding" and late["percent"] == 92 and late["updated"] == int(START)


def test_a_riot_rate_limit_is_waited_out_then_the_call_is_retried():
    # given
    from synergy.worker import riot_call

    answers = [RateLimited(7.0), RateLimited(0.2), "ok"]
    naps = []

    def call():
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    # when
    found = riot_call(call, _Accounts(), pause=naps.append)

    # then
    assert found == "ok" and naps == [7.0, 1.0, 1.25]


def test_an_interrupted_evaluation_queues_itself_again_from_the_start():
    # given
    table = _Table()
    table.put(EVAL, "me", {"status": RUNNING, "step": "embedding", "percent": 92, "riot_id": "Me#NA1"})
    sent = []

    # when
    requeue("me", table, "https://queue", lambda **message: sent.append(message))

    # then
    state = table.get(EVAL, "me")
    assert state["status"] == "requested" and state["step"] == "queued" and state["percent"] == 0 and state["riot_id"] == "Me#NA1"
    assert sent == [{"QueueUrl": "https://queue", "MessageBody": '{"puuid": "me"}'}]


def test_the_profile_notes_the_newest_game_that_was_read():
    # given
    from synergy.evaluate import profile_of

    seat = {"puuid": "p", "championName": "Garen", "teamPosition": "TOP", "win": True}
    games = [
        ("m1", {"info": {"gameCreation": 1_790_000_000_000, "gameDuration": 1_800_000, "participants": [seat]}}, {}),
        ("m2", {"info": {"gameCreation": 1_790_100_000_000, "gameDuration": 1500, "gameEndTimestamp": 1_790_101_500_000, "participants": [seat]}}, {}),
    ]
    result = {"games": 2, "position": ["TOP"], "seats": [2]}

    # when
    profile = profile_of("p", games, {"gameName": "P", "tagLine": "NA1"}, [], result)

    # then
    assert profile["latest_game"] == 1_790_101_500 and profile["main_champion"] == "Garen" and profile["games"] == 2
