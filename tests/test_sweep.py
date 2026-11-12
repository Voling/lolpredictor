import json

from synergy.api.accounts import LocalTable
from synergy.api.evaluate import EVAL, FAILED, JOBS, READY, REQUESTED, RUNNING, SYSTEM
from synergy.api.sweep import dead_letters, redrive, sweep
from synergy.config import Settings

START = 1_790_000_000.0
DAY = 86_400
TODAY = "2026-09-21"


def _settings():
    return Settings(compute_retries=3, compute_stall_seconds=600, sweep_seconds=1800)


def _player(table, puuid, **changes):
    table.put(EVAL, puuid, {"status": RUNNING, "step": "fetching", "riot_id": f"{puuid}#NA1", "started": int(START), "updated": int(START), "expires": int(START) + 14 * DAY, "by": "user-1", **changes})


def _players(table):
    _player(table, "quiet")
    table.put(JOBS, "quiet", {"job": "job-1", "started": int(START)})
    _player(table, "fetching", updated=int(START) + 1000)
    _player(table, "computing", task="task-a", attempts=1, updated=int(START) + 1100)
    table.put(JOBS, "computing", {"job": "job-2", "started": int(START), "sent": int(START) + 500})
    _player(table, "asked", started=int(START) + 1000, updated=int(START) + 1100)
    table.put(JOBS, "asked", {"job": "job-0", "started": int(START), "sent": int(START) + 10})
    _player(table, "spent", status=REQUESTED, step="queued", attempts=3)
    _player(table, "done", status=READY)
    table.add("user#user-1", f"evals#{TODAY}", 2, 3, int(START) + 2 * DAY)
    table.add(EVAL, f"day#{TODAY}", 2, 20, int(START) + 2 * DAY)
    return table


def test_the_sweep_asks_again_for_quiet_evaluations_leaves_live_ones_and_fails_those_out_of_tries():
    # given
    table, asked = _players(LocalTable()), []

    # when
    tally = sweep(table, asked.append, _settings(), clock=lambda: START + 1801)

    # then
    quiet, computing, spent = (table.get(EVAL, puuid) for puuid in ("quiet", "computing", "spent"))
    assert tally == {"live": 2, "asked_again": 2, "failed": 1} and asked == ["computing", "quiet"]
    assert (quiet["status"], quiet["step"], quiet["attempts"], quiet["updated"], quiet["by"]) == (REQUESTED, "queued", 1, int(START) + 1801, "user-1")
    assert (computing["status"], computing["attempts"], "task" in computing) == (REQUESTED, 2, False)
    assert table.get(JOBS, "quiet") is None and table.get(JOBS, "computing") is None and table.get(JOBS, "asked")["job"] == "job-0"
    assert (spent["status"], spent["kind"], spent["error"]) == (FAILED, SYSTEM, "We hit a problem pulling spent#NA1's games. Try again.")
    assert table.get("user#user-1", f"evals#{TODAY}")["used"] == 1 and table.get(EVAL, f"day#{TODAY}")["used"] == 1
    assert table.get(EVAL, "fetching")["status"] == RUNNING and table.get(EVAL, "done")["status"] == READY


def test_the_sweep_writes_nothing_when_a_live_job_moves_between_its_read_and_its_write():
    # given
    class _Racing(LocalTable):
        def every(self, pk):
            found = super().every(pk)
            self.raise_to(EVAL, "quiet", "updated", int(START) + 1800)
            self.put(EVAL, "spent", {**self.get(EVAL, "spent"), "status": RUNNING, "task": "task-b", "updated": int(START) + 1800})
            return found

    table, asked = _players(_Racing()), []

    # when
    tally = sweep(table, asked.append, _settings(), clock=lambda: START + 1801)

    # then
    assert tally["failed"] == 0 and asked == ["computing"] and table.get(JOBS, "quiet")["job"] == "job-1"
    assert table.get(EVAL, "quiet")["status"] == RUNNING and table.get(EVAL, "spent")["status"] == RUNNING
    assert table.get("user#user-1", f"evals#{TODAY}")["used"] == 2


class _Sqs:
    def __init__(self, queues):
        self.queues = {url: [{"ReceiptHandle": f"{url}-{index}", "Body": body, "hidden": False} for index, body in enumerate(bodies)] for url, bodies in queues.items()}
        self.waits = []

    def get_queue_attributes(self, QueueUrl, AttributeNames):
        return {"Attributes": {"ApproximateNumberOfMessages": str(len(self.queues[QueueUrl]))}}

    def receive_message(self, QueueUrl, MaxNumberOfMessages, VisibilityTimeout, WaitTimeSeconds):
        self.waits.append(VisibilityTimeout)
        found = [message for message in self.queues[QueueUrl] if not message["hidden"]][:MaxNumberOfMessages]
        for message in found:
            message["hidden"] = VisibilityTimeout > 0
        return {"Messages": [{"ReceiptHandle": message["ReceiptHandle"], "Body": message["Body"]} for message in found]}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.queues[QueueUrl] = [message for message in self.queues[QueueUrl] if message["ReceiptHandle"] != ReceiptHandle]


def test_the_dead_letter_report_counts_and_samples_each_queue_without_moving_anything():
    # given
    sqs = _Sqs({"history": [json.dumps({"puuid": f"p{index}"}) for index in range(7)], "downloads": ['{"puuid": "a"}'], "evaluate": []})

    # when
    report = dead_letters(sqs, {"history": "history", "downloads": "downloads", "evaluate": "evaluate"})

    # then
    assert {kind: (found["messages"], len(found["samples"])) for kind, found in report.items()} == {"history": (7, 5), "downloads": (1, 1), "evaluate": (0, 0)}
    assert report["downloads"]["samples"] == ['{"puuid": "a"}'] and set(sqs.waits) == {0} and len(sqs.queues["history"]) == 7


def test_a_redrive_asks_once_for_each_player_resets_their_tries_charges_nobody_and_keeps_what_it_cannot_read():
    # given
    table, asked = LocalTable(), []
    _player(table, "a", status=FAILED, kind=SYSTEM, error="We hit a problem pulling a#NA1's games. Try again.", attempts=3, refunded=1)
    _player(table, "b", attempts=2, by="visitor#x")
    table.put(JOBS, "a", {"job": "job-1", "started": int(START)})
    table.add("user#user-1", f"evals#{TODAY}", 1, 3, int(START) + 2 * DAY)
    game = json.dumps({"puuid": "a", "job": "job-1", "match_id": "NA1_1"})
    sqs = _Sqs({"history": ['{"puuid": "a"}', '{"puuid": "b"}'], "downloads": [game, game, "not a job"], "evaluate": ['{"puuid": "c"}']})

    # when
    found = redrive(sqs, {"history": "history", "downloads": "downloads", "evaluate": "evaluate"}, table, asked.append, clock=lambda: START + 600)

    # then
    a, b, c = (table.get(EVAL, puuid) for puuid in "abc")
    assert found == {"players": 3, "deleted": 5, "kept": 1} and asked == ["a", "b", "c"]
    assert (a["status"], a["step"], a["updated"]) == (REQUESTED, "queued", int(START) + 600) and not {"attempts", "kind", "refunded", "error", "by"} & set(a)
    assert (b["status"], b["by"], "attempts" in b) == (REQUESTED, "visitor#x", False) and (c["status"], c["started"]) == (REQUESTED, int(START) + 600)
    assert table.get(JOBS, "a") is None and table.get("user#user-1", f"evals#{TODAY}")["used"] == 1
    assert [message["Body"] for message in sqs.queues["downloads"]] == ["not a job"] and sqs.queues["history"] == sqs.queues["evaluate"] == []
