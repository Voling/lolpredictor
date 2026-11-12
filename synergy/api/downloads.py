import gzip
import itertools
import json
import logging
import math
import queue
import random
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from .accounts import DAY, TIMEOUT, LinkError, LocalTable, NotFound, RateLimited, RiotRefused
from .evaluate import EVAL, FAILED, GAMES, JOBS, PLAYER, REQUESTED, STORE, SYSTEM, SYSTEM_PROBLEM, Progress, fail_evaluation
from .limiter import patiently, riot_client

logger = logging.getLogger(__name__)
ATTEMPTS = "attempts"
RANKED = "ranked"
GAME_PREFIX = "games/"
BATCH = 10
JOB_DAYS = 2
RETRY_SECONDS = 2.0
LONGEST_WAIT = 900
SHORTEST_WAIT = 5
HISTORY_CACHE = "history_cache"
NO_GAMES = "Riot returned no ranked games for this player."
NONE_FETCHED = "We couldn't pull any of this player's games from Riot. Try again later."
STORE_FULL = "Our game store is full right now. Try again tomorrow."


class NoGames(ValueError):
    pass


def backoff(attempt: int, rand=random.random, cap: int = LONGEST_WAIT) -> int:
    high = max(SHORTEST_WAIT, min(cap, 10 * 2 ** int(attempt)))
    return math.ceil(SHORTEST_WAIT + (high - SHORTEST_WAIT) * rand())


def deferral(retry_after: float, rand=random.random) -> int:
    return max(1, math.ceil(min(LONGEST_WAIT, float(retry_after) * (1 + 0.25 * rand()))))


def packed(match: dict, timeline: dict) -> bytes:
    return gzip.compress(json.dumps({"match": match, "timeline": timeline}).encode("utf-8"))


class GameStore:
    def __init__(self, client, bucket: str, table, cap_bytes: int):
        self.client, self.bucket, self.table, self.cap_bytes = client, bucket, table, cap_bytes

    def key(self, match_id: str) -> str:
        return f"{GAME_PREFIX}{match_id}.json.gz"

    def has(self, match_id: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(match_id))
        except self.client.exceptions.ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def read(self, match_id: str) -> dict | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key(match_id))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        return json.loads(gzip.decompress(body))

    def write(self, match_id: str, match: dict, timeline: dict) -> int:
        body = packed(match, timeline)
        self.client.put_object(Bucket=self.bucket, Key=self.key(match_id), Body=body, ContentType="application/json", ContentEncoding="gzip")
        self.table.add(STORE, "bytes", len(body), 1 << 62, int(time.time()) + 3 * DAY)
        return len(body)


class FolderStore:
    def __init__(self, root: Path):
        self.root, self.written, self.lock = Path(root), 0, threading.Lock()

    def path(self, match_id: str) -> Path:
        return self.root / f"{GAME_PREFIX}{match_id}.json.gz"

    def has(self, match_id: str) -> bool:
        return self.path(match_id).exists()

    def write(self, match_id: str, match: dict, timeline: dict) -> int:
        body, target = packed(match, timeline), self.path(match_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.with_name(f"{target.name}.{threading.get_ident()}.part")
        scratch.write_bytes(body)
        scratch.replace(target)
        with self.lock:
            self.written += 1
        return len(body)


def orchestrate(puuid: str, riot, table, store, send, launch, store_full=lambda: False, clock=time.time, new_job=lambda: uuid.uuid4().hex) -> dict | None:
    progress = Progress(table, puuid, clock)
    progress.step("fetching", 0, GAMES)
    ids = [str(match_id) for match_id in riot.match_ids(puuid, RANKED, GAMES)]
    if not ids:
        fail_evaluation(table, puuid, NO_GAMES, PLAYER, clock)
        return None
    missing = [match_id for match_id in ids if not store.has(match_id)]
    if missing and store_full():
        logger.warning("the game store is full, keeping the %s cached games before %s", ids.index(missing[0]), missing[0])
        ids, missing = ids[: ids.index(missing[0])], []
        if not ids:
            fail_evaluation(table, puuid, STORE_FULL, SYSTEM, clock)
            return None
    now = int(clock())
    job = {"job": new_job(), "ids": json.dumps(ids), "total": len(ids), "expected": len(missing), "started": now, "expires": now + JOB_DAYS * DAY}
    table.put(JOBS, puuid, job)
    progress.step("fetching", len(ids) - len(missing), len(ids))
    if not missing:
        launch(puuid)
    for start in range(0, len(missing), BATCH):
        send([{"puuid": puuid, "job": job["job"], "match_id": match_id} for match_id in missing[start : start + BATCH]])
    logger.info("%s has %s games, %s to download", puuid[:8], len(ids), len(missing))
    return job


def download(message: dict, message_id: str, riot, table, store, launch, clock=time.time) -> None:
    puuid, job, match_id = message["puuid"], message["job"], message["match_id"]
    current = table.get(JOBS, puuid)
    if not current or current.get("job") != job:
        logger.info("dropping %s, its job is gone", match_id)
        return
    dead = match_id in (current.get("missing") or ())
    if not dead and not store.has(match_id):
        try:
            match, timeline = riot.game(match_id)
        except RiotRefused:
            dead = True
        else:
            store.write(match_id, match, timeline)
        if dead:
            logger.warning("skipping %s, Riot would not send it", match_id)
    _counted(table, puuid, job, match_id, dead, message_id, launch, clock)


def _counted(table, puuid: str, job: str, match_id: str, dead: bool, message_id: str, launch, clock) -> None:
    found = table.collect(JOBS, puuid, ["done", "missing"] if dead else ["done"], match_id, {"job": job})
    if found is None:
        return
    done, total, expected = len(found.get("done") or ()), int(found["total"]), int(found["expected"])
    table.raise_to(EVAL, puuid, "done", total - expected + done, {"step": "fetching"}, also={"updated": int(clock())})
    if done >= expected and table.flag(JOBS, puuid, "launched", message_id, unless="sent"):
        if len(found.get("missing") or ()) >= total:
            fail_evaluation(table, puuid, NONE_FETCHED, PLAYER, clock)
            return
        launch(puuid)
        table.raise_to(JOBS, puuid, "sent", int(clock()), {"job": job})


def abandon_history(table, body: dict, error: Exception, clock=time.time) -> None:
    if isinstance(error, NotFound):
        fail_evaluation(table, body["puuid"], NO_GAMES, PLAYER, clock)
    else:
        fail_evaluation(table, body["puuid"], SYSTEM_PROBLEM, SYSTEM, clock)


def abandon_download(table, body: dict, message_id: str, error: Exception, launch, clock=time.time) -> None:
    puuid, job, match_id = body["puuid"], body.get("job"), body.get("match_id")
    current = table.get(JOBS, puuid)
    if not current or current.get("job") != job:
        return
    if isinstance(error, LinkError) and match_id:
        logger.warning("skipping %s, Riot kept failing", match_id)
        _counted(table, puuid, job, match_id, True, message_id, launch, clock)
    else:
        fail_evaluation(table, puuid, SYSTEM_PROBLEM, SYSTEM, clock)


def _body(record: dict) -> dict | None:
    try:
        body = json.loads(record["body"])
    except (KeyError, TypeError, ValueError):
        return None
    return body if isinstance(body, dict) and isinstance(body.get("puuid"), str) else None


def _retried(record: dict, body: dict, error: Exception, give_up, retries: int, table, clock, rand) -> int | None:
    message_id = record["messageId"]
    try:
        tries = retries if isinstance(error, RiotRefused) else table.add(ATTEMPTS, message_id, 1, 1 << 62, int(clock()) + JOB_DAYS * DAY) or retries
    except Exception:
        logger.exception("could not count a try for %s", message_id)
        return 0
    if tries < retries:
        logger.warning("try %s of %s failed for %s", tries, retries, message_id, exc_info=error)
        return backoff(tries, rand)
    logger.error("giving up on %s after %s tries", message_id, tries, exc_info=error)
    try:
        give_up(body, message_id, error)
    except Exception:
        logger.exception("could not give up on %s", message_id)
        return 0
    return None


def run_batch(records: list[dict], work, give_up, retries: int, table, defer, clock=time.time, rand=random.random) -> dict:
    failed, busy = [], None
    for record in records:
        message_id, body = record["messageId"], _body(record)
        if body is None:
            logger.error("dropping %s, it is not a job", message_id)
            continue
        if busy is not None:
            wait = deferral(busy.retry_after, rand)
        else:
            try:
                work(body, message_id)
                continue
            except RateLimited as limited:
                busy, wait = limited, deferral(limited.retry_after, rand)
                logger.info("riot is busy for %.1fs, %s and the rest of its batch wait", limited.retry_after, message_id)
            except Exception as error:
                wait = _retried(record, body, error, give_up, retries, table, clock, rand)
                if wait is None:
                    continue
        failed.append({"itemIdentifier": message_id})
        if wait:
            try:
                defer(record, wait)
            except Exception:
                logger.exception("could not hold %s back for %ss", message_id, wait)
    return {"batchItemFailures": failed}


def pull_history(riot_id: str, workers: int, settings, transport=None, clock=time.monotonic) -> dict:
    import httpx

    started, tally, lock = clock(), Counter(), threading.Lock()
    limits, table, store = LocalTable(), LocalTable(), FolderStore(settings.data_dir / HISTORY_CACHE)
    finished, launched = threading.Event(), threading.Event()
    waiting, numbers, riots = queue.PriorityQueue(), itertools.count(), []

    def count(response) -> None:
        with lock:
            tally["calls"] += 1
            tally["rate_limited"] += int(response.status_code == 429)

    def client():
        riots.append(riot_client(settings, limits, httpx.Client(timeout=TIMEOUT, transport=transport, event_hooks={"response": [count]})))
        return riots[-1]

    def send(bodies: list[dict]) -> None:
        for body in bodies:
            waiting.put((clock(), next(numbers), body, 1))

    def launch(puuid: str) -> None:
        launched.set()
        finished.set()

    def work(riot) -> None:
        while not finished.is_set():
            try:
                due, number, body, tries = waiting.get(timeout=0.2)
            except queue.Empty:
                continue
            if finished.wait(max(0.0, due - clock())):
                return
            try:
                download(body, f"local-{number}", riot, table, store, launch)
            except RateLimited as limited:
                waiting.put((clock() + limited.retry_after, number, body, tries))
            except Exception as error:
                logger.exception("could not download %s", body["match_id"])
                if tries >= settings.game_retries:
                    abandon_download(table, body, f"local-{number}", error, launch)
                else:
                    waiting.put((clock() + RETRY_SECONDS, number, body, tries + 1))

    riot = client()
    account = patiently(lambda: riot.account(riot_id))
    puuid, name = account["puuid"], f"{account.get('gameName')}#{account.get('tagLine')}"
    table.put(EVAL, puuid, {"status": REQUESTED, "riot_id": name})
    job = patiently(lambda: orchestrate(puuid, riot, table, store, send, launch)) or {}
    if not job:
        finished.set()
    threads = [threading.Thread(target=work, args=(client(),), daemon=True) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    while not finished.wait(0.5):
        if (table.get(EVAL, puuid) or {}).get("status") == FAILED:
            finished.set()
    for thread in threads:
        thread.join()
    for each in riots:
        each.client.close()
    return {
        "riot_id": name,
        "games": int(job.get("total", 0)),
        "cached": int(job.get("total", 0)) - int(job.get("expected", 0)),
        "downloaded": store.written,
        "calls": tally["calls"],
        "rate_limited": tally["rate_limited"],
        "seconds": round(clock() - started, 1),
        "complete": launched.is_set(),
        "error": (table.get(EVAL, puuid) or {}).get("error"),
        "folder": str(store.root),
    }
