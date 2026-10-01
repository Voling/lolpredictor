import gzip
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from .api.accounts import DAY, Accounts, DynamoTable, LinkError, RateLimited, RiotAccounts, ssm_key
from .api.artifacts import bucket_of, current_run
from .api.evaluate import EVAL, FAILED, GAMES, NAMES, READY, REQUESTED, RUNNING, name_key, store_size
from .config import Settings, get_settings
from .evaluate import BUNDLE, Bundle, Evaluator, profile_of, save_result

logger = logging.getLogger(__name__)
RANKED = "ranked"
GAME_PREFIX = "games/"
WEIGHTS = {"fetching": (0, 50), "loading": (50, 60), "features": (60, 70), "positions": (70, 78), "reactions": (78, 82), "tendencies": (82, 86), "habits": (86, 88), "movement": (88, 92), "embedding": (92, 97), "scoring": (97, 100)}
BUDGET_WAIT = 180.0
PACE_SECONDS = 1.25
RETRIES = 6


class Progress:
    def __init__(self, table, puuid: str, clock=time.time):
        self.table, self.puuid, self.clock = table, puuid, clock
        self.state = table.get(EVAL, puuid) or {}

    def write(self, **changes) -> None:
        self.state = {**self.state, **changes, "updated": int(self.clock())}
        self.table.put(EVAL, self.puuid, self.state)

    def step(self, name: str, done: int = 0, total: int = 0) -> None:
        low, high = WEIGHTS.get(name, (0, 0))
        share = (done / total) if total else 0.0
        self.write(status=RUNNING, step=name, done=int(done), total=int(total), percent=int(low + (high - low) * share))


class GameStore:
    def __init__(self, client, bucket: str, table, cap_bytes: int):
        self.client, self.bucket, self.table, self.cap_bytes = client, bucket, table, cap_bytes

    def key(self, match_id: str) -> str:
        return f"{GAME_PREFIX}{match_id}.json.gz"

    def read(self, match_id: str) -> dict | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key(match_id))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        return json.loads(gzip.decompress(body))

    def write(self, match_id: str, match: dict, timeline: dict) -> int:
        body = gzip.compress(json.dumps({"match": match, "timeline": timeline}).encode("utf-8"))
        self.client.put_object(Bucket=self.bucket, Key=self.key(match_id), Body=body, ContentType="application/json", ContentEncoding="gzip")
        self.table.add("evaluated", "bytes", len(body), 1 << 62, int(time.time()) + 3 * DAY)
        return len(body)


def wait_for_budget(accounts: Accounts, calls: int, clock=time.time, pause=time.sleep) -> None:
    started = clock()
    while not accounts.riot_calls(calls):
        if clock() - started > BUDGET_WAIT:
            raise LinkError("Riot is busy right now. Try again in a few minutes.")
        pause(2.0)


def riot_call(call, accounts: Accounts, pause=time.sleep):
    for attempt in range(RETRIES):
        wait_for_budget(accounts, 1, pause=pause)
        try:
            found = call()
        except RateLimited as limited:
            logger.info("riot asked for %.0fs of patience", limited.retry_after)
            pause(min(max(limited.retry_after, 1.0), 120.0))
            continue
        pause(PACE_SECONDS)
        return found
    raise LinkError("Riot is busy right now. Try again in a few minutes.")


def fetch_games(riot: RiotAccounts, accounts: Accounts, store: GameStore, puuid: str, progress, store_full=lambda: False, pause=time.sleep) -> list[tuple[str, dict, dict]]:
    match_ids = riot_call(lambda: riot.match_ids(puuid, RANKED, GAMES), accounts, pause)
    games = []
    for number, match_id in enumerate(match_ids, start=1):
        found = store.read(match_id)
        if found is None:
            if store_full():
                logger.warning("the game store is full, stopping at %s games", len(games))
                break
            match = riot_call(lambda: riot.match(match_id), accounts, pause)
            timeline = riot_call(lambda: riot.timeline(match_id), accounts, pause)
            store.write(match_id, match, timeline)
            found = {"match": match, "timeline": timeline}
        games.append((match_id, found["match"], found["timeline"]))
        progress("fetching", number, len(match_ids))
    return games


def fetch_bundle(client, bucket: str, target: Path) -> Path:
    run = current_run(client, bucket)
    prefix = f"runs/{run}/{BUNDLE}/"
    target.mkdir(parents=True, exist_ok=True)
    page = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [item["Key"] for item in page.get("Contents", [])]
    if not keys:
        raise FileNotFoundError(f"run {run} has no evaluator bundle under {prefix}")
    for key in keys:
        client.download_file(bucket, key, str(target / key.rsplit("/", 1)[-1]))
    return target


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


def evaluate_player(puuid: str, settings: Settings | None = None, scratch: Path | None = None) -> dict:
    import boto3

    settings = settings or get_settings()
    scratch = scratch or Path(settings.data_dir)
    table = DynamoTable(settings.accounts_table)
    progress = Progress(table, puuid)
    key = ssm_key(settings.riot_key_parameter) if settings.riot_key_parameter else (lambda: settings.riot_api_key)
    riot = RiotAccounts(key, settings.platform, settings.region)
    accounts = Accounts(table, riot, settings.daily_duos, settings.link_attempts, settings.riot_budget)
    s3 = boto3.client("s3")
    store = GameStore(s3, bucket_of(settings.evaluated_store), table, settings.evaluated_cap_gb * (1 << 30))
    try:
        progress.step("fetching", 0, GAMES)
        games = fetch_games(riot, accounts, store, puuid, progress.step, lambda: store_size(settings, table) >= store.cap_bytes)
        if not games:
            raise ValueError("Riot returned no ranked games for this player")
        bundle = Bundle(fetch_bundle(s3, bucket_of(settings.model_store), scratch / "bundle"))
        wait_for_database(settings)
        local = Settings(data_dir=scratch / "data", database_url=settings.database_url, timescale=False)
        result = Evaluator(bundle, local, progress.step).run(puuid, games)
        account = riot_call(lambda: riot.account_by_puuid(puuid), accounts)
        league = riot_call(lambda: riot.league(puuid), accounts)
        profile = profile_of(puuid, games, account, league, result)
        vectors, summary = save_result(scratch / "out", result, profile)
        s3.upload_file(str(vectors), store.bucket, f"vectors/{puuid}.npz")
        s3.upload_file(str(summary), store.bucket, f"profiles/{puuid}.json")
        now = int(time.time())
        riot_id = f"{profile.get('game_name')}#{profile.get('tag_line')}"
        table.put(NAMES, name_key(riot_id), {"puuid": puuid, "expires": now + 14 * DAY})
        progress.write(status=READY, step="ready", percent=100, riot_id=riot_id, games=result["games"], expires=now + 14 * DAY)
        logger.info("evaluated %s from %s games", riot_id, result["games"])
        return profile
    except Interrupted:
        raise
    except Exception as exc:
        logger.exception("evaluation failed for %s", puuid)
        progress.write(status=FAILED, step="failed", error=str(exc)[:200])
        raise
    finally:
        shutil.rmtree(scratch / "data", ignore_errors=True)


class Interrupted(SystemExit):
    pass


def requeue(puuid: str, table, queue_url: str, send) -> None:
    state = table.get(EVAL, puuid) or {}
    table.put(EVAL, puuid, {**state, "status": REQUESTED, "step": "queued", "percent": 0, "started": int(time.time())})
    send(QueueUrl=queue_url, MessageBody=json.dumps({"puuid": puuid}))
    logger.warning("interrupted while evaluating %s, queued again", puuid[:8])


def on_interruption(puuid: str, settings: Settings):
    def handle(signum, frame):
        import boto3

        if settings.evaluate_queue and settings.accounts_table:
            requeue(puuid, DynamoTable(settings.accounts_table), settings.evaluate_queue, boto3.client("sqs").send_message)
        raise Interrupted(143)

    return handle


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    puuid = os.environ.get("EVALUATE_PUUID", "").strip()
    if not puuid:
        print("EVALUATE_PUUID is not set", file=sys.stderr)
        return 2
    settings = get_settings()
    signal.signal(signal.SIGTERM, on_interruption(puuid, settings))
    try:
        evaluate_player(puuid, settings)
    except Interrupted as stop:
        return int(stop.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
