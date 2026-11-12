import gzip
import json

import pytest

from synergy.api.accounts import LocalTable, NotFound
from synergy.api.downloads import JOBS, NO_GAMES, NONE_FETCHED, GameStore, NoGames
from synergy.api.evaluate import EVAL, FAILED, PLAYER, READY, REQUESTED, RUNNING, SYSTEM, Progress, Superseded
from synergy.config import Settings
from synergy.worker import Interrupted, claim, fetch_games, run_claimed

START = 1_790_000_000.0
DAY = 86_400


class _Missing(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

        ClientError = _Missing

    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if Key == "games/denied.json.gz":
            raise _Missing("403")
        if Key not in self.objects:
            raise _Missing("404")
        return {}

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
    def __init__(self, ids, dead=()):
        self.ids, self.dead, self.calls = ids, set(dead), []

    def match_ids(self, puuid, kind, count=100):
        self.calls.append(("ids", kind, count))
        return list(self.ids)

    def game(self, match_id):
        self.calls.append(("game", match_id))
        if match_id in self.dead:
            raise NotFound("Riot doesn't know that Riot ID. Check the name and tag.")
        return {"metadata": {"matchId": match_id}, "info": {"participants": []}}, {"info": {"frames": []}}


def _cached(s3, match_id, frames):
    s3.objects[f"games/{match_id}.json.gz"] = gzip.compress(json.dumps({"match": {"metadata": {"matchId": match_id}, "info": {"participants": []}}, "timeline": {"info": {"frames": frames}}}).encode())


def test_without_a_job_the_evaluator_lists_the_games_itself_and_stores_new_ones_compressed():
    # given
    table, s3 = LocalTable(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    _cached(s3, "NA1_1", [1])
    riot, seen = _Riot(["NA1_1", "NA1_2"]), []

    # when
    games = fetch_games(riot, store, table, "me", lambda step, done, total: seen.append((step, done, total)), pause=lambda seconds: None)

    # then
    assert [match_id for match_id, _, _ in games] == ["NA1_1", "NA1_2"] and games[0][2] == {"info": {"frames": [1]}}
    assert riot.calls == [("ids", "ranked", 50), ("game", "NA1_2")]
    assert "games/NA1_2.json.gz" in s3.objects and table.get("evaluated", "bytes")["used"] == len(s3.objects["games/NA1_2.json.gz"])
    assert seen == [("fetching", 0, 50), ("fetching", 1, 2), ("fetching", 2, 2)]


def test_the_evaluator_reads_the_match_ids_from_the_job_and_finds_every_game_cached():
    # given
    table, s3 = LocalTable(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    for match_id in ("NA1_2", "NA1_1"):
        _cached(s3, match_id, [match_id])
    table.put(JOBS, "me", {"job": "job-1", "ids": json.dumps(["NA1_2", "NA1_1"]), "total": 2, "expected": 0})
    riot, seen = _Riot(["NA1_9"]), []

    # when
    games = fetch_games(riot, store, table, "me", lambda *parts: seen.append(parts), pause=lambda seconds: None)

    # then
    assert [match_id for match_id, _, _ in games] == ["NA1_2", "NA1_1"] and games[0][2] == {"info": {"frames": ["NA1_2"]}}
    assert riot.calls == [] and seen == []


def test_a_full_store_keeps_cached_games_and_stops_pulling_new_ones():
    # given
    table, s3 = LocalTable(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    _cached(s3, "NA1_1", [])
    riot = _Riot(["NA1_1", "NA1_2", "NA1_3"])

    # when
    games = fetch_games(riot, store, table, "me", lambda *parts: None, store_full=lambda: True, pause=lambda seconds: None)

    # then
    assert [match_id for match_id, _, _ in games] == ["NA1_1"] and ("game", "NA1_2") not in riot.calls


def test_the_evaluator_runs_on_the_games_it_has_and_skips_games_riot_lost():
    # given
    table, s3 = LocalTable(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    _cached(s3, "NA1_1", [1])
    _cached(s3, "NA1_4", [4])
    table.put(JOBS, "me", {"job": "job-1", "ids": json.dumps(["NA1_1", "NA1_2", "NA1_3", "NA1_4"]), "total": 4, "expected": 2, "missing": ["NA1_2"]})
    riot = _Riot([], dead={"NA1_3"})

    # when
    games = fetch_games(riot, store, table, "me", lambda *parts: None, pause=lambda seconds: None)

    # then
    assert [match_id for match_id, _, _ in games] == ["NA1_1", "NA1_4"] and riot.calls == [("game", "NA1_3")]


def test_the_evaluator_fails_plainly_only_when_none_of_the_games_could_be_fetched():
    # given
    table, s3 = LocalTable(), _S3()
    store = GameStore(s3, "bucket", table, 10 << 30)
    table.put(JOBS, "me", {"job": "job-1", "ids": json.dumps(["NA1_1", "NA1_2"]), "total": 2, "expected": 2, "missing": ["NA1_1"]})
    riot = _Riot([], dead={"NA1_2"})

    # when
    with pytest.raises(ValueError) as none:
        fetch_games(riot, store, table, "me", lambda *parts: None, pause=lambda seconds: None)

    # then
    assert str(none.value) == NONE_FETCHED and riot.calls == [("game", "NA1_2")]


def test_the_game_store_knows_a_cached_game_from_a_head_request_and_raises_other_errors():
    # given
    s3 = _S3()
    store = GameStore(s3, "bucket", LocalTable(), 10 << 30)
    _cached(s3, "NA1_1", [])

    # when
    found = (store.has("NA1_1"), store.has("NA1_2"))
    with pytest.raises(_Missing):
        store.has("denied")

    # then
    assert found == (True, False)


def _settings():
    return Settings(evaluate_queue="https://queue", compute_retries=3, compute_stall_seconds=600)


def _waiting(**changes):
    table = LocalTable()
    table.put(EVAL, "me", {"status": RUNNING, "step": "fetching", "done": 50, "total": 50, "riot_id": "Me#NA1", "started": int(START), "updated": int(START), "expires": int(START) + 14 * DAY, "by": "user-1", **changes})
    table.add("user#user-1", "evals#2026-09-21", 1, 3, int(START) + 2 * DAY)
    table.add(EVAL, "day#2026-09-21", 1, 20, int(START) + 2 * DAY)
    return table


def _run(table, task, compute, sent, at=START + 60, rand=lambda: 0.5):
    return run_claimed("me", table, task, _settings(), compute, lambda **message: sent.append(message), clock=lambda: at, rand=rand, beat=lambda *parts: (lambda: None))


def _failing(error):
    def compute(progress):
        raise error

    return compute


def test_the_first_task_claims_the_work_and_a_duplicate_launch_exits_without_computing():
    # given
    table, ran, sent = _waiting(), [], []

    def compute(progress):
        ran.append(progress.owner)
        return "profile"

    # when
    first = _run(table, "task-a", compute, sent)
    second = _run(table, "task-b", compute, sent, at=START + 120)
    Progress(table, "me", owner="task-a").write(status=READY, step="ready")
    late = _run(table, "task-c", compute, sent, at=START + 9000)

    # then
    assert (first, second, late) == ("profile", None, None) and ran == ["task-a"] and sent == []
    assert (table.get(EVAL, "me")["task"], table.get(EVAL, "me")["attempts"]) == ("task-a", 1)


def test_a_task_whose_heartbeat_went_quiet_is_taken_over_and_its_late_writes_are_refused():
    # given
    table = _waiting(task="task-old", attempts=1)
    old = Progress(table, "me", owner="task-old")

    # when
    early = claim(table, "me", "task-new", 600, clock=lambda: START + 599)
    taken = claim(table, "me", "task-new", 600, clock=lambda: START + 600)
    with pytest.raises(Superseded):
        old.step("embedding")

    # then
    state = table.get(EVAL, "me")
    assert early is None and taken["task"] == "task-new" and (state["task"], state["attempts"], state["updated"], state["status"]) == ("task-new", 2, int(START) + 600, RUNNING)
    assert state["step"] == "fetching"


def test_a_passing_failure_goes_back_on_the_queue_after_a_backoff_and_keeps_the_job():
    # given
    table, sent = _waiting(), []
    table.put(JOBS, "me", {"job": "job-1", "ids": "[]", "started": int(START), "sent": int(START)})

    # when
    _run(table, "task-a", _failing(RuntimeError("Riot answered 503")), sent)

    # then
    state = table.get(EVAL, "me")
    assert (state["status"], state["step"], state["attempts"], state["updated"], "task" in state) == (REQUESTED, "queued", 1, int(START) + 60, False)
    assert sent == [{"QueueUrl": "https://queue", "MessageBody": '{"puuid": "me"}', "DelaySeconds": 13}] and table.get(JOBS, "me")["job"] == "job-1"


def test_a_spot_interruption_takes_the_same_path_and_counts_a_try():
    # given
    table, sent = _waiting(attempts=1), []

    # when
    with pytest.raises(Interrupted):
        _run(table, "task-a", _failing(Interrupted(143)), sent, rand=lambda: 1.0)

    # then
    state = table.get(EVAL, "me")
    assert (state["status"], state["attempts"]) == (REQUESTED, 2) and sent[0]["DelaySeconds"] == 40


def test_the_last_try_fails_as_ours_and_gives_the_slot_back():
    # given
    table, sent = _waiting(attempts=2), []

    # when
    _run(table, "task-a", _failing(RuntimeError("out of disk")), sent)

    # then
    state = table.get(EVAL, "me")
    assert (state["status"], state["kind"], state["attempts"], state["refunded"]) == (FAILED, SYSTEM, 3, 1) and sent == []
    assert state["error"] == "We hit a problem pulling Me#NA1's games. Try again." and "task" not in state
    assert table.get("user#user-1", "evals#2026-09-21")["used"] == 0 and table.get(EVAL, "day#2026-09-21")["used"] == 0


def test_a_player_without_usable_games_fails_as_theirs_at_once_and_keeps_the_slot_spent():
    # given
    table, sent = _waiting(), []

    # when
    _run(table, "task-a", _failing(NoGames(NO_GAMES)), sent)

    # then
    state = table.get(EVAL, "me")
    assert (state["status"], state["kind"], state["error"], state["attempts"]) == (FAILED, PLAYER, NO_GAMES, 1) and sent == []
    assert "refunded" not in state and table.get("user#user-1", "evals#2026-09-21")["used"] == 1


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
