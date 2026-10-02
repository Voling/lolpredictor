import time
from datetime import datetime, timezone

from ..ml.evaluated import FRESH_SECONDS
from .accounts import DAY, LinkError, QuotaExceeded, riot_id_of

EVAL = "eval"
NAMES = "evalname"
STORE = "evaluated"
REQUESTED, RUNNING, READY, FAILED = "requested", "running", "ready", "failed"
GAMES = 50
STALE_SECONDS = 1800
GIB = 1024**3


def name_key(riot_id: str) -> str:
    return " ".join(str(riot_id).split()).lower()


def store_size(settings, table) -> int:
    import datetime

    import boto3

    bucket = settings.evaluated_store.removeprefix("s3://").split("/", 1)[0]
    now = datetime.datetime.now(datetime.timezone.utc)
    found = boto3.client("cloudwatch").get_metric_statistics(
        Namespace="AWS/S3",
        MetricName="BucketSizeBytes",
        Dimensions=[{"Name": "BucketName", "Value": bucket}, {"Name": "StorageType", "Value": "StandardStorage"}],
        StartTime=now - datetime.timedelta(days=3),
        EndTime=now,
        Period=86400,
        Statistics=["Maximum"],
    )
    measured = max((point["Maximum"] for point in found.get("Datapoints", [])), default=0.0)
    recent = table.get(STORE, "bytes") or {}
    return int(measured) + int(recent.get("used", 0))


def shape(state: dict | None, riot_id: str, message: str | None = None) -> dict:
    state = state or {}
    status = state.get("status", "refused")
    return {
        "riot_id": state.get("riot_id", riot_id),
        "status": status,
        "step": state.get("step", status),
        "done": int(state.get("done", 0) or 0),
        "total": int(state.get("total", 0) or 0),
        "percent": int(state.get("percent", 0) or 0),
        "message": message or state.get("error"),
    }


class Evaluations:
    def __init__(self, table, riot, budget, launch=None, store_size=lambda: 0, cap_bytes: int = 10 * GIB, daily: int = 20, per_user: int = 3, keep_days: int = 14, clock=time.time):
        self.table, self.riot, self.budget, self.launch, self.store_size = table, riot, budget, launch, store_size
        self.cap_bytes, self.daily, self.per_user, self.keep_days, self.clock = cap_bytes, daily, per_user, keep_days, clock

    def day(self) -> str:
        return f"{datetime.fromtimestamp(self.clock(), timezone.utc):%Y-%m-%d}"

    def status(self, puuid: str) -> dict | None:
        return self.table.get(EVAL, puuid)

    def lookup(self, riot_id: str) -> dict | None:
        found = self.table.get(NAMES, name_key(riot_id))
        return self.status(found["puuid"]) if found else None

    def refuse(self, user: str, riot_id: str):
        state = self.lookup(riot_id)
        now = int(self.clock())
        if state:
            name, age = state.get("riot_id", riot_id), now - int(state.get("started", 0))
            if state["status"] in (REQUESTED, RUNNING) and age < STALE_SECONDS:
                raise LinkError(f"We're still pulling {name}'s games. Try again in a few minutes.")
            if state["status"] == READY and now < int(state.get("expires", 0)):
                raise LinkError(f"We have {name}'s games but the model hasn't picked them up yet. Try again in a minute.")
            if state["status"] == FAILED and age < DAY:
                raise LinkError(f"We couldn't pull {name}'s games. Try again tomorrow.")
        self.request(user, riot_id)

    def refresh(self, user: str, riot_id: str, latest_game: int | None = None):
        state = self.lookup(riot_id)
        now = int(self.clock())
        name = state.get("riot_id", riot_id) if state else riot_id
        if state and state["status"] in (REQUESTED, RUNNING) and now - int(state.get("started", 0)) < STALE_SECONDS:
            raise LinkError(f"We're still pulling {name}'s games. Try again in a few minutes.")
        if latest_game is not None and now - int(latest_game) < FRESH_SECONDS:
            when = datetime.fromtimestamp(int(latest_game) + FRESH_SECONDS, timezone.utc)
            raise LinkError(f"We already have {name}'s games from the last two weeks. New games can be pulled after {when:%B} {when.day}.")
        self.request(user, riot_id, again=True)

    def _give_back(self, user: str, expires: int, day: bool) -> None:
        self.table.add(f"user#{user}", f"evals#{self.day()}", -1, self.per_user + 1, expires)
        if day:
            self.table.add(EVAL, f"day#{self.day()}", -1, self.daily + 1, expires)

    def request(self, user: str, riot_id: str, again: bool = False):
        riot_id = riot_id_of(riot_id)
        if self.store_size() >= self.cap_bytes:
            raise LinkError("Our game store is full right now. Try again tomorrow.")
        expires = int(self.clock()) + 2 * DAY
        if self.table.add(f"user#{user}", f"evals#{self.day()}", 1, self.per_user, expires) is None:
            raise QuotaExceeded(f"You can ask for {self.per_user} new players a day. Try again tomorrow.")
        if self.table.add(EVAL, f"day#{self.day()}", 1, self.daily, expires) is None:
            self._give_back(user, expires, day=False)
            raise LinkError("We've pulled as many new players as we can today. Try again tomorrow.")
        if not self.budget(1):
            self._give_back(user, expires, day=True)
            raise LinkError("Riot is busy right now. Try again in a few minutes.")
        try:
            account = self.riot.account(riot_id)
        except Exception:
            self._give_back(user, expires, day=True)
            raise
        puuid = account["puuid"]
        name = f"{account.get('gameName', riot_id.partition('#')[0])}#{account.get('tagLine', riot_id.partition('#')[2])}"
        now = int(self.clock())
        self.table.put(NAMES, name_key(name), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        if name_key(name) != name_key(riot_id):
            self.table.put(NAMES, name_key(riot_id), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        state = {"status": REQUESTED, "riot_id": name, "started": now, "expires": now + self.keep_days * DAY, "by": user}
        self.table.put(EVAL, puuid, state)
        if self.launch is not None:
            try:
                self.launch(puuid)
            except Exception:
                self.table.put(EVAL, puuid, {**state, "status": FAILED})
                raise LinkError(f"We couldn't start pulling {name}'s games. Try again later.")
        if again:
            raise LinkError(f"We're pulling {name}'s newest games now. Check back in about 10 minutes.")
        raise LinkError(f"We don't know {name} yet. We're pulling their last {GAMES} ranked games now. Check back in about 10 minutes.")
