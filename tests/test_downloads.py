import json

import httpx
import pytest

from synergy.api.accounts import UNREACHABLE, LinkError, LocalTable, NotFound, RateLimited, RiotAccounts, RiotRefused
from synergy.api.downloads import ATTEMPTS, JOBS, NO_GAMES, NONE_FETCHED, STORE_FULL, abandon_download, abandon_history, backoff, download, orchestrate, pull_history, run_batch
from synergy.api.evaluate import EVAL, FAILED, PLAYER, REQUESTED, RUNNING, SYSTEM, shape
from synergy.api.limiter import LIMITS, RiotLimiter
from synergy.config import Settings

START = 1_790_000_040.0


class _Riot:
    def __init__(self, ids, refuse=None, broken=None):
        self.ids, self.refuse, self.broken, self.calls = ids, refuse, dict(broken or {}), []

    def match_ids(self, puuid, kind, count=100):
        if self.refuse is not None:
            raise self.refuse
        self.calls.append(("ids", kind, count))
        return list(self.ids)

    def game(self, match_id):
        if self.refuse is not None:
            raise self.refuse
        self.calls.append(("game", match_id))
        if match_id in self.broken:
            raise self.broken[match_id]
        return {"metadata": {"matchId": match_id}}, {"info": {"frames": []}}


class _Store:
    def __init__(self, cached=(), broken=None):
        self.games, self.broken = {match_id: ({}, {}) for match_id in cached}, broken

    def has(self, match_id):
        return match_id in self.games

    def write(self, match_id, match, timeline):
        if self.broken is not None:
            raise self.broken
        self.games[match_id] = (match, timeline)
        return 1


def _started(ids, cached=(), job="job-1", store=None):
    table, store, riot, sent, launched = LocalTable(), store or _Store(cached), _Riot(ids), [], []
    table.put(EVAL, "me", {"status": REQUESTED, "riot_id": "Me#NA1", "started": int(START), "updated": int(START)})
    orchestrate("me", riot, table, store, sent.append, launched.append, clock=lambda: START, new_job=lambda: job)
    return table, store, riot, {body["match_id"]: body for batch in sent for body in batch}, launched


def _record(body, message_id):
    return {"messageId": message_id, "receiptHandle": f"receipt-{message_id}", "body": json.dumps(body)}


def _downloads(records, riot, table, store, launched, held, retries=5, rand=lambda: 0.5):
    return run_batch(
        records,
        lambda body, message_id: download(body, message_id, riot, table, store, launched.append, clock=lambda: START),
        lambda body, message_id, error: abandon_download(table, body, message_id, error, launched.append, clock=lambda: START),
        retries,
        table,
        lambda record, seconds: held.append((record["messageId"], seconds)),
        clock=lambda: START,
        rand=rand,
    )


def _histories(records, riot, table, held, retries=5, rand=lambda: 0.5):
    return run_batch(
        records,
        lambda body, message_id: orchestrate(body["puuid"], riot, table, _Store(), lambda bodies: None, lambda puuid: None, clock=lambda: START),
        lambda body, message_id, error: abandon_history(table, body, error, clock=lambda: START),
        retries,
        table,
        lambda record, seconds: held.append((record["messageId"], seconds)),
        clock=lambda: START,
        rand=rand,
    )


def _tries(table):
    return {sk: found["used"] for (pk, sk), found in table.items.items() if pk == ATTEMPTS}


def test_the_orchestrator_queues_each_missing_game_in_batches_of_ten_and_skips_cached_ones():
    # given
    ids = [f"NA1_{index}" for index in range(25)]
    table, store, riot, sent, launched = LocalTable(), _Store(["NA1_0", "NA1_7"]), _Riot(ids), [], []
    table.put(EVAL, "me", {"status": REQUESTED, "riot_id": "Me#NA1"})

    # when
    job = orchestrate("me", riot, table, store, sent.append, launched.append, clock=lambda: START, new_job=lambda: "job-1")

    # then
    bodies = [body for batch in sent for body in batch]
    assert [len(batch) for batch in sent] == [10, 10, 3] and launched == [] and riot.calls == [("ids", "ranked", 50)]
    assert [body["match_id"] for body in bodies] == [match_id for match_id in ids if match_id not in ("NA1_0", "NA1_7")]
    assert {(body["puuid"], body["job"]) for body in bodies} == {("me", "job-1")}
    stored = table.get(JOBS, "me")
    assert stored == job and json.loads(stored["ids"]) == ids and (stored["total"], stored["expected"]) == (25, 23)
    state = table.get(EVAL, "me")
    assert (state["status"], state["step"], state["done"], state["total"], state["riot_id"]) == (RUNNING, "fetching", 2, 25, "Me#NA1")


def test_the_evaluation_is_launched_at_once_when_every_game_is_cached():
    # given
    ids = ["NA1_1", "NA1_2"]

    # when
    table, store, riot, sent, launched = _started(ids, cached=ids)

    # then
    assert launched == ["me"] and sent == {} and table.get(JOBS, "me")["expected"] == 0
    assert shape(table.get(EVAL, "me"), "Me#NA1")["percent"] == 50


def test_a_full_store_keeps_only_the_cached_games_before_the_first_missing_one():
    # given
    table, launched = LocalTable(), []
    ids = ["NA1_1", "NA1_2", "NA1_3", "NA1_4"]

    # when
    orchestrate("me", _Riot(ids), table, _Store(["NA1_1", "NA1_2", "NA1_4"]), launched.append, launched.append, store_full=lambda: True)
    orchestrate("you", _Riot(["NA1_3", "NA1_1"]), table, _Store(["NA1_1"]), launched.append, launched.append, store_full=lambda: True)

    # then
    assert json.loads(table.get(JOBS, "me")["ids"]) == ["NA1_1", "NA1_2"] and launched == ["me"]
    you = table.get(EVAL, "you")
    assert table.get(JOBS, "you") is None and (you["status"], you["error"], you["kind"]) == (FAILED, STORE_FULL, SYSTEM)


def test_a_player_without_ranked_games_fails_without_a_job_or_an_evaluation():
    # given
    table, launched = LocalTable(), []

    # when
    found = orchestrate("me", _Riot([]), table, _Store(), launched.append, launched.append)

    # then
    state = table.get(EVAL, "me")
    assert found is None and launched == [] and table.get(JOBS, "me") is None
    assert (state["status"], state["step"], state["error"], state["kind"]) == (FAILED, "failed", NO_GAMES, PLAYER)


def test_retried_downloads_count_once_and_exactly_one_evaluation_is_launched():
    # given
    ids = ["NA1_1", "NA1_2", "NA1_3"]
    table, store, riot, messages, launched = _started(ids)

    # when
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_2"], "m-2", riot, table, store, launched.append)
    middle = shape(table.get(EVAL, "me"), "Me#NA1")
    for message_id, match_id in (("m-3", "NA1_3"), ("m-3", "NA1_3"), ("m-2", "NA1_2"), ("m-1", "NA1_1")):
        download(messages[match_id], message_id, riot, table, store, launched.append)

    # then
    job = table.get(JOBS, "me")
    assert launched == ["me"] and [call for call in riot.calls if call[0] == "game"] == [("game", "NA1_1"), ("game", "NA1_2"), ("game", "NA1_3")]
    assert job["done"] == ids and job["launched"] == "m-3" and job["sent"] > 0
    assert (middle["step"], middle["done"], middle["total"], middle["percent"]) == ("fetching", 2, 3, 33)
    assert shape(table.get(EVAL, "me"), "Me#NA1")["done"] == 3


def test_each_counted_game_bumps_the_heartbeat_of_the_evaluation():
    # given
    table, store, riot, messages, launched = _started(["NA1_1", "NA1_2"])
    before = table.get(EVAL, "me")["updated"]

    # when
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append, clock=lambda: START + 90)

    # then
    state = table.get(EVAL, "me")
    assert before == int(START) and (state["done"], state["updated"]) == (1, int(START) + 90)


def test_a_launch_that_failed_before_sending_is_sent_when_its_message_comes_back():
    # given
    table, store, riot, messages, launched = _started(["NA1_1"])

    def broken(puuid):
        raise RuntimeError("the queue is down")

    # when
    with pytest.raises(RuntimeError):
        download(messages["NA1_1"], "m-1", riot, table, store, broken)
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)

    # then
    assert launched == ["me"] and riot.calls.count(("game", "NA1_1")) == 1


def test_a_game_riot_refuses_counts_as_done_without_it_and_is_never_asked_for_again():
    # given
    table, store, riot, messages, launched = _started(["NA1_1", "NA1_2", "NA1_3"])
    riot.broken = {"NA1_1": NotFound("Riot doesn't know that Riot ID. Check the name and tag."), "NA1_2": RiotRefused("Riot refused that request. Check the name and tag.")}

    # when
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_2"], "m-2", riot, table, store, launched.append)
    download(messages["NA1_3"], "m-3", riot, table, store, launched.append)

    # then
    job = table.get(JOBS, "me")
    assert job["missing"] == ["NA1_1", "NA1_2"] and job["done"] == ["NA1_1", "NA1_2", "NA1_3"] and launched == ["me"]
    assert riot.calls.count(("game", "NA1_1")) == 1 and "NA1_1" not in store.games


def test_a_job_whose_every_game_is_dead_fails_plainly_without_launching_an_evaluation():
    # given
    table, store, riot, messages, launched = _started(["NA1_1", "NA1_2"])
    riot.broken = {match_id: NotFound("Riot doesn't know that Riot ID. Check the name and tag.") for match_id in ("NA1_1", "NA1_2")}

    # when
    download(messages["NA1_1"], "m-1", riot, table, store, launched.append)
    download(messages["NA1_2"], "m-2", riot, table, store, launched.append)

    # then
    state = table.get(EVAL, "me")
    assert launched == [] and (state["status"], state["error"], state["kind"]) == (FAILED, NONE_FETCHED, PLAYER) and table.get(JOBS, "me") is None


def test_a_message_from_a_replaced_job_is_dropped_without_asking_riot():
    # given
    table, store, riot, old, launched = _started(["NA1_1", "NA1_2"], job="job-1")
    orchestrate("me", riot, table, store, lambda bodies: None, launched.append, new_job=lambda: "job-2")

    # when
    download(old["NA1_1"], "m-1", riot, table, store, launched.append)

    # then
    assert ("game", "NA1_1") not in riot.calls and "done" not in table.get(JOBS, "me") and launched == []


def test_a_rate_limited_batch_waits_out_retry_after_with_a_quarter_of_jitter_and_counts_no_try():
    # given
    table, store, _, messages, launched = _started(["NA1_1", "NA1_2", "NA1_3"])
    limits, asked, held = LocalTable(), [], []
    limits.put(LIMITS, "americas#120", {"used": 90, "start": START - 80})
    riot = RiotAccounts(lambda: "key", "na1", "americas", RiotLimiter(limits, "20/1,100/120", clock=lambda: START), httpx.Client(transport=httpx.MockTransport(lambda request: asked.append(request) or httpx.Response(200, json={}))))
    records = [_record(messages[match_id], f"m-{index}") for index, match_id in enumerate(("NA1_1", "NA1_2", "NA1_3"))]

    # when
    answer = _downloads(records, riot, table, store, launched, held)
    capped = []
    _downloads([records[0]], _Riot([], refuse=RateLimited(850.0)), table, store, launched, capped)

    # then
    assert answer == {"batchItemFailures": [{"itemIdentifier": "m-0"}, {"itemIdentifier": "m-1"}, {"itemIdentifier": "m-2"}]}
    assert held == [("m-0", 45), ("m-1", 45), ("m-2", 45)] and capped == [("m-0", 900)]
    assert asked == [] and store.games == {} and _tries(table) == {} and table.get(EVAL, "me")["status"] == RUNNING


def test_a_rejected_key_holds_the_batch_back_for_the_breaker_without_counting_a_try():
    # given
    table, store, _, messages, launched = _started(["NA1_1", "NA1_2"])
    asked, held = [], []

    def answer(request):
        asked.append(request.url.path)
        return httpx.Response(401)

    riot = RiotAccounts(lambda: "dead-key", "na1", "americas", RiotLimiter(LocalTable(), "20/1,100/120", clock=lambda: START), httpx.Client(transport=httpx.MockTransport(answer)))
    records = [_record(messages["NA1_1"], "m-1"), _record(messages["NA1_2"], "m-2")]

    # when
    first = _downloads(records, riot, table, store, launched, held, rand=lambda: 0.4)
    again = _downloads(records, riot, table, store, launched, held, rand=lambda: 0.0)

    # then
    assert first == again == {"batchItemFailures": [{"itemIdentifier": "m-1"}, {"itemIdentifier": "m-2"}]}
    assert held == [("m-1", 660), ("m-2", 660), ("m-1", 600), ("m-2", 600)]
    assert asked == ["/lol/match/v5/matches/NA1_1"] and _tries(table) == {} and table.get(EVAL, "me")["status"] == RUNNING


def test_a_failing_list_counts_each_try_backs_off_with_full_jitter_and_then_fails_as_ours():
    # given
    table, held, rolls = LocalTable(), [], iter([0.0, 1.0, 0.5, 0.25, 0.9])
    table.put(EVAL, "me", {"status": REQUESTED, "riot_id": "Me#NA1", "started": int(START)})
    riot = _Riot([], refuse=LinkError(UNREACHABLE))
    record = _record({"puuid": "me"}, "h-1")

    # when
    answers = [_histories([record], riot, table, held, rand=lambda: next(rolls)) for _ in range(5)]

    # then
    state = table.get(EVAL, "me")
    assert [answer["batchItemFailures"] for answer in answers] == [[{"itemIdentifier": "h-1"}]] * 4 + [[]]
    assert held == [("h-1", 5), ("h-1", 40), ("h-1", 43), ("h-1", 44)] and _tries(table) == {"h-1": 5}
    assert (state["status"], state["kind"], state["error"]) == (FAILED, SYSTEM, "We hit a problem pulling Me#NA1's games. Try again.")
    assert backoff(7, lambda: 1.0) == 900


def test_a_permanent_refusal_fails_at_once_without_a_try_and_a_message_that_is_not_a_job_is_dropped():
    # given
    table, held = LocalTable(), []
    for puuid in ("me", "you"):
        table.put(EVAL, puuid, {"status": REQUESTED, "riot_id": f"{puuid}#NA1", "started": int(START)})

    # when
    unknown = _histories([_record({"puuid": "me"}, "h-1"), {"messageId": "h-2", "body": "not a job"}], _Riot([], refuse=NotFound("Riot doesn't know that Riot ID. Check the name and tag.")), table, held)
    refused = _histories([_record({"puuid": "you"}, "h-3")], _Riot([], refuse=RiotRefused("Riot refused that request. Check the name and tag.")), table, held)

    # then
    me, you = table.get(EVAL, "me"), table.get(EVAL, "you")
    assert unknown == refused == {"batchItemFailures": []} and held == [] and _tries(table) == {}
    assert (me["status"], me["kind"], me["error"]) == (FAILED, PLAYER, NO_GAMES)
    assert (you["status"], you["kind"], you["error"]) == (FAILED, SYSTEM, "We hit a problem pulling you#NA1's games. Try again.")


def test_a_game_riot_keeps_failing_is_skipped_after_its_tries_and_the_rest_is_evaluated():
    # given
    table, store, riot, messages, launched = _started(["NA1_1", "NA1_2"], cached=["NA1_2"])
    riot.broken = {"NA1_1": LinkError(UNREACHABLE)}
    held = []

    # when
    answers = [_downloads([_record(messages["NA1_1"], "m-1")], riot, table, store, launched, held, retries=3) for _ in range(3)]

    # then
    assert [answer["batchItemFailures"] for answer in answers] == [[{"itemIdentifier": "m-1"}]] * 2 + [[]]
    assert riot.calls.count(("game", "NA1_1")) == 3 and [seconds for _, seconds in held] == [13, 23] and launched == ["me"]
    assert table.get(JOBS, "me")["missing"] == ["NA1_1"] and table.get(EVAL, "me")["status"] == RUNNING


def test_our_own_failure_while_downloading_fails_the_evaluation_as_ours_after_its_tries():
    # given
    table, store, riot, messages, launched = _started(["NA1_1", "NA1_2"], store=_Store(broken=RuntimeError("S3 is down")))
    held = []

    # when
    answers = [_downloads([_record(messages["NA1_1"], "m-1")], riot, table, store, launched, held, retries=2) for _ in range(2)]
    late = _downloads([_record(messages["NA1_2"], "m-2")], riot, table, store, launched, held, retries=2)

    # then
    state = table.get(EVAL, "me")
    assert [answer["batchItemFailures"] for answer in answers] == [[{"itemIdentifier": "m-1"}], []] and late == {"batchItemFailures": []}
    assert (state["status"], state["kind"]) == (FAILED, SYSTEM) and table.get(JOBS, "me") is None and launched == []


def test_the_local_runner_downloads_a_history_into_its_folder_under_the_shared_limits(tmp_path):
    # given
    games, limited = ["NA1_1", "NA1_2", "NA1_3"], []

    def answer(request):
        path = request.url.path
        if "/by-riot-id/" in path:
            return httpx.Response(200, json={"puuid": "p", "gameName": "Me", "tagLine": "NA1"})
        if path.endswith("/ids"):
            return httpx.Response(200, json=games)
        if path.endswith("/timeline"):
            return httpx.Response(200, json={"info": {"frames": []}})
        if path.endswith("NA1_2") and not limited:
            limited.append(path)
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"metadata": {"matchId": path.rsplit("/", 1)[-1]}})

    settings = Settings(data_dir=tmp_path, riot_api_key="key", riot_key_parameter="", riot_limits="20/1,100/120", game_retries=3)
    (tmp_path / "history_cache" / "games").mkdir(parents=True)
    (tmp_path / "history_cache" / "games" / "NA1_3.json.gz").write_bytes(b"cached")

    # when
    report = pull_history("Me#NA1", 2, settings, transport=httpx.MockTransport(answer))

    # then
    assert report["complete"] is True and report["error"] is None and (report["games"], report["cached"], report["downloaded"]) == (3, 1, 2)
    assert report["calls"] == 7 and report["rate_limited"] == 1
    assert sorted(path.name for path in (tmp_path / "history_cache" / "games").iterdir()) == ["NA1_1.json.gz", "NA1_2.json.gz", "NA1_3.json.gz"]
