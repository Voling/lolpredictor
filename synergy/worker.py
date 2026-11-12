import json
import logging
import os
import random
import shutil
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

from .api.accounts import DAY, DynamoTable, LinkError, NotFound, RateLimited, RiotRefused
from .api.artifacts import bucket_of, current_run
from .api.downloads import JOBS, LONGEST_WAIT, NO_GAMES, NONE_FETCHED, RANKED, GameStore, NoGames, backoff
from .api.evaluate import EVAL, GAMES, GIB, NAMES, PLAYER, READY, REQUESTED, RUNNING, SYSTEM, SYSTEM_PROBLEM, Progress, Superseded, fail_evaluation, name_key, queued_state, since, store_size, unchanged
from .api.limiter import patiently, riot_client
from .config import Settings, get_settings
from .evaluate import BUNDLE, Bundle, Evaluator, profile_of, save_result

logger = logging.getLogger(__name__)
LOCAL_TASK = "local"
HEARTBEAT_SECONDS = 60


def fetch_games(riot, store: GameStore, table, puuid: str, progress, store_full=lambda: False, pause=time.sleep) -> list[tuple[str, dict, dict]]:
    job = table.get(JOBS, puuid)
    if job:
        match_ids, missing = json.loads(job["ids"]), set(job.get("missing") or ())
    else:
        progress("fetching", 0, GAMES)
        match_ids, missing = patiently(lambda: riot.match_ids(puuid, RANKED, GAMES), pause), set()
    if not match_ids:
        raise NoGames(NO_GAMES)
    games = []
    for number, match_id in enumerate(match_ids, start=1):
        if match_id not in missing:
            found = store.read(match_id)
            if found is None:
                if store_full():
                    logger.warning("the game store is full, stopping at %s games", len(games))
                    break
                found = _fetched(riot, store, match_id, pause)
            if found is not None:
                games.append((match_id, found["match"], found["timeline"]))
        if not job:
            progress("fetching", number, len(match_ids))
    if not games:
        raise NoGames(NONE_FETCHED)
    return games


def _fetched(riot, store: GameStore, match_id: str, pause) -> dict | None:
    try:
        match, timeline = patiently(lambda: riot.game(match_id), pause)
    except RateLimited:
        raise
    except LinkError as problem:
        logger.warning("skipping %s: %s", match_id, problem)
        return None
    store.write(match_id, match, timeline)
    return {"match": match, "timeline": timeline}


def fetch_bundle(client, bucket: str, target: Path) -> tuple[Path, str]:
    run = current_run(client, bucket)
    prefix = f"runs/{run}/{BUNDLE}/"
    target.mkdir(parents=True, exist_ok=True)
    page = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [item["Key"] for item in page.get("Contents", [])]
    if not keys:
        raise FileNotFoundError(f"run {run} has no evaluator bundle under {prefix}")
    for key in keys:
        client.download_file(bucket, key, str(target / key.rsplit("/", 1)[-1]))
    return target, run


def wait_for_database(settings: Settings, pause=time.sleep, tries: int = 60) -> None:
    import psycopg

    for attempt in range(tries):
        try:
            psycopg.connect(settings.database_url, connect_timeout=5).close()
            return
        except psycopg.Error:
            if attempt == tries - 1:
                raise
            pause(2.0)


class Interrupted(SystemExit):
    pass


def task_arn(environ=os.environ) -> str:
    uri = environ.get("ECS_CONTAINER_METADATA_URI_V4")
    if not uri:
        return LOCAL_TASK
    import httpx

    try:
        return str(httpx.get(f"{uri}/task", timeout=2.0).json()["TaskARN"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        logger.warning("could not read this task's ARN, using a random one")
        return f"task-{uuid.uuid4().hex}"


def claim(table, puuid: str, task: str, stall_seconds: int, clock=time.time) -> dict | None:
    state = table.get(EVAL, puuid)
    if not state or state.get("status") not in (REQUESTED, RUNNING):
        return None
    now = int(clock())
    if state.get("task") and now - since(state) < stall_seconds:
        return None
    claimed = {**{name: value for name, value in state.items() if name not in ("pk", "sk")}, "status": RUNNING, "task": task, "attempts": int(state.get("attempts", 0) or 0) + 1, "updated": now}
    return claimed if table.put_if(EVAL, puuid, claimed, unchanged(state)) else None


def settle(table, puuid: str, task: str, error: BaseException, settings: Settings, send, clock=time.time, rand=random.random) -> None:
    owner = {"task": task}
    if isinstance(error, (NoGames, NotFound)):
        fail_evaluation(table, puuid, str(error), PLAYER, clock, owner)
        return
    state = table.get(EVAL, puuid) or {}
    attempts = int(state.get("attempts", 0) or 0)
    if isinstance(error, RiotRefused) or attempts >= settings.compute_retries or not settings.evaluate_queue:
        fail_evaluation(table, puuid, SYSTEM_PROBLEM, SYSTEM, clock, owner)
        return
    delay = backoff(attempts, rand, min(LONGEST_WAIT, settings.compute_stall_seconds // 2))
    if table.put_if(EVAL, puuid, queued_state(state, int(clock())), owner):
        send(QueueUrl=settings.evaluate_queue, MessageBody=json.dumps({"puuid": puuid}), DelaySeconds=delay)
        logger.warning("try %s of %s failed for %s, trying again in %ss", attempts, settings.compute_retries, puuid[:8], delay)


def _beat(table, puuid: str, task: str, stop: threading.Event, clock, beats: int) -> None:
    for _ in range(beats):
        if stop.wait(HEARTBEAT_SECONDS):
            return
        try:
            table.raise_to(EVAL, puuid, "updated", int(clock()), {"task": task})
        except Exception:
            logger.warning("could not mark %s as alive", puuid[:8], exc_info=True)


def heartbeat(table, puuid: str, task: str, clock, beats: int):
    stop = threading.Event()
    threading.Thread(target=_beat, args=(table, puuid, task, stop, clock, beats), daemon=True).start()
    return stop.set


def run_claimed(puuid: str, table, task: str, settings: Settings, compute, send, clock=time.time, rand=random.random, beat=heartbeat):
    if claim(table, puuid, task, settings.compute_stall_seconds, clock) is None:
        logger.info("%s is done or another task has it", puuid[:8])
        return None
    stop = beat(table, puuid, task, clock, 3 * settings.compute_stall_seconds // HEARTBEAT_SECONDS)
    try:
        return compute(Progress(table, puuid, clock, owner=task))
    except Superseded:
        logger.warning("another task took over %s", puuid[:8])
    except Interrupted as interrupted:
        logger.warning("interrupted while evaluating %s", puuid[:8])
        settle(table, puuid, task, interrupted, settings, send, clock, rand)
        raise
    except Exception as error:
        logger.exception("evaluation failed for %s", puuid[:8])
        settle(table, puuid, task, error, settings, send, clock, rand)
    finally:
        stop()
    return None


def evaluate_player(puuid: str, settings: Settings | None = None, scratch: Path | None = None) -> dict | None:
    import boto3

    settings = settings or get_settings()
    scratch = scratch or Path(settings.data_dir)
    table = DynamoTable(settings.accounts_table)
    riot = riot_client(settings, table)
    s3 = boto3.client("s3")
    store = GameStore(s3, bucket_of(settings.evaluated_store), table, settings.evaluated_cap_gb * GIB)

    def compute(progress: Progress) -> dict:
        games = fetch_games(riot, store, table, puuid, progress.step, lambda: store_size(settings, table) >= store.cap_bytes)
        folder, run = fetch_bundle(s3, bucket_of(settings.model_store), scratch / "bundle")
        bundle = Bundle(folder)
        wait_for_database(settings)
        local = Settings(data_dir=scratch / "data", database_url=settings.database_url, timescale=False)
        result = {**Evaluator(bundle, local, progress.step).run(puuid, games), "run": run}
        account = patiently(lambda: riot.account_by_puuid(puuid))
        league = patiently(lambda: riot.league(puuid))
        now = int(time.time())
        profile = {**profile_of(puuid, games, account, league, result), "evaluated_at": now}
        vectors, summary = save_result(scratch / "out", result, profile)
        s3.upload_file(str(vectors), store.bucket, f"vectors/{puuid}.npz")
        s3.upload_file(str(summary), store.bucket, f"profiles/{puuid}.json")
        riot_id = f"{profile.get('game_name')}#{profile.get('tag_line')}"
        table.put(NAMES, name_key(riot_id), {"puuid": puuid, "expires": now + 14 * DAY})
        progress.write(status=READY, step="ready", riot_id=riot_id, games=result["games"], expires=now + 14 * DAY, finished=now, latest_game=profile.get("latest_game"), run=run)
        logger.info("evaluated %s from %s games", riot_id, result["games"])
        return profile

    try:
        return run_claimed(puuid, table, task_arn(), settings, compute, boto3.client("sqs").send_message)
    finally:
        shutil.rmtree(scratch / "data", ignore_errors=True)


def interrupt(signum, frame):
    raise Interrupted(143)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    puuid = os.environ.get("EVALUATE_PUUID", "").strip()
    if not puuid:
        print("EVALUATE_PUUID is not set", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, interrupt)
    try:
        evaluate_player(puuid, get_settings())
    except Interrupted as stop:
        return int(stop.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
