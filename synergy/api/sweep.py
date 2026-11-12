import json
import logging
import time

from .accounts import DAY
from .evaluate import EVAL, JOBS, KEEP_DAYS, REQUESTED, RUNNING, SYSTEM, SYSTEM_PROBLEM, fail_evaluation, queued_state, since, unchanged

logger = logging.getLogger(__name__)
DEAD_QUEUES = ("history", "downloads", "evaluate")
SAMPLES = 5
HOLD_SECONDS = 60


def _current_job(table, puuid: str, state: dict) -> dict:
    job = table.get(JOBS, puuid) or {}
    return job if int(job.get("started", 0) or 0) >= int(state.get("started", 0) or 0) else {}


def sweep(table, request, settings, clock=time.time) -> dict:
    now, tally = int(clock()), {"live": 0, "asked_again": 0, "failed": 0}
    for state in table.every(EVAL):
        if state.get("status") not in (REQUESTED, RUNNING):
            continue
        puuid = str(state["sk"])
        try:
            limit = settings.compute_stall_seconds if _current_job(table, puuid, state).get("sent") else settings.sweep_seconds
            quiet, attempts = now - since(state), int(state.get("attempts", 0) or 0)
            if quiet < limit:
                tally["live"] += 1
            elif attempts >= settings.compute_retries:
                if fail_evaluation(table, puuid, SYSTEM_PROBLEM, SYSTEM, clock, unchanged(state)):
                    tally["failed"] += 1
                    logger.warning("%s went quiet for %ss after %s tries, failed it", puuid[:8], quiet, attempts)
            elif table.put_if(EVAL, puuid, queued_state(state, now, attempts=attempts + 1), unchanged(state)):
                table.delete(JOBS, puuid)
                request(puuid)
                tally["asked_again"] += 1
                logger.warning("%s went quiet for %ss, asked for it again", puuid[:8], quiet)
        except Exception:
            logger.exception("could not sweep %s", puuid[:8])
    return tally


def dead_letters(sqs, queues: dict[str, str], samples: int = SAMPLES) -> dict:
    report = {}
    for kind, url in queues.items():
        count = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages"])["Attributes"]["ApproximateNumberOfMessages"]
        found = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=samples, VisibilityTimeout=0, WaitTimeSeconds=1).get("Messages", [])
        report[kind] = {"messages": int(count), "samples": [message["Body"] for message in found]}
    return report


def _puuid(body: str) -> str | None:
    try:
        found = json.loads(body)
    except (TypeError, ValueError):
        return None
    return found["puuid"] if isinstance(found, dict) and isinstance(found.get("puuid"), str) else None


def reopen(table, puuid: str, clock=time.time) -> None:
    state, now = table.get(EVAL, puuid) or {}, int(clock())
    drop = ("attempts", "refunded", "by") if state.get("refunded") else ("attempts", "refunded")
    table.put(EVAL, puuid, queued_state(state, now, drop, started=state.get("started", now), expires=state.get("expires", now + KEEP_DAYS * DAY)))
    table.delete(JOBS, puuid)


def redrive(sqs, queues: dict[str, str], table, request, clock=time.time) -> dict:
    asked, moved, kept = set(), 0, 0
    for url in queues.values():
        while True:
            found = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, VisibilityTimeout=HOLD_SECONDS, WaitTimeSeconds=1).get("Messages", [])
            if not found:
                break
            for message in found:
                puuid = _puuid(message["Body"])
                if puuid is None:
                    kept += 1
                    continue
                if puuid not in asked:
                    reopen(table, puuid, clock)
                    request(puuid)
                    asked.add(puuid)
                sqs.delete_message(QueueUrl=url, ReceiptHandle=message["ReceiptHandle"])
                moved += 1
    return {"players": len(asked), "deleted": moved, "kept": kept}
