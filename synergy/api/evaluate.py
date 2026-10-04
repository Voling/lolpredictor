import time
from datetime import datetime, timezone

from ..ml.evaluated import FRESH_SECONDS
from .accounts import DAY, VISITOR, LinkError, QuotaExceeded, riot_id_of

EVAL = "eval"
NAMES = "evalname"
STORE = "evaluated"
REQUESTED, RUNNING, READY, FAILED = "requested", "running", "ready", "failed"
UNDER_WAY = (REQUESTED, RUNNING, READY)
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
    def __init__(
        self,
        table,
        riot,
        budget,
        launch=None,
        store_size=lambda: 0,
        cap_bytes: int = 10 * GIB,
        daily: int = 20,
        per_user: int = 3,
        per_visitor: int = 3,
        visitors_daily: int = 150,
        visitor_budget=None,
        keep_days: int = 14,
        clock=time.time,
    ):
        self.table, self.riot, self.budget, self.launch, self.store_size = table, riot, budget, launch, store_size
        self.cap_bytes, self.daily, self.per_user, self.keep_days, self.clock = cap_bytes, daily, per_user, keep_days, clock
        self.per_visitor, self.visitors_daily, self.visitor_budget = per_visitor, visitors_daily, visitor_budget or budget

    def day(self) -> str:
        return f"{datetime.fromtimestamp(self.clock(), timezone.utc):%Y-%m-%d}"

    def status(self, puuid: str) -> dict | None:
        return self.table.get(EVAL, puuid)

    def lookup(self, riot_id: str) -> dict | None:
        found = self.table.get(NAMES, name_key(riot_id))
        return self.status(found["puuid"]) if found else None

    def _held(self, state: dict | None, riot_id: str, again: bool) -> None:
        if not state:
            return
        now = int(self.clock())
        name, age = state.get("riot_id", riot_id), now - int(state.get("started", 0))
        if state["status"] in (REQUESTED, RUNNING) and age < STALE_SECONDS:
            raise LinkError(f"We're still pulling {name}'s games. Try again in a few minutes.")
        if again:
            return
        if state["status"] == READY and now < int(state.get("expires", 0)):
            raise LinkError(f"We have {name}'s games but the model hasn't picked them up yet. Try again in a minute.")
        if state["status"] == FAILED and age < DAY:
            raise LinkError(f"We couldn't pull {name}'s games. Try again tomorrow.")

    def refuse(self, user: str, riot_id: str):
        self._held(self.lookup(riot_id), riot_id, again=False)
        self.request(user, riot_id)

    def refresh(self, user: str, riot_id: str, latest_game: int | None = None):
        state = self.lookup(riot_id)
        name = state.get("riot_id", riot_id) if state else riot_id
        self._held(state, riot_id, again=True)
        if latest_game is not None and int(self.clock()) - int(latest_game) < FRESH_SECONDS:
            when = datetime.fromtimestamp(int(latest_game) + FRESH_SECONDS, timezone.utc)
            raise LinkError(f"We already have {name}'s games from the last two weeks. New games can be pulled after {when:%B} {when.day}.")
        self.request(user, riot_id, again=True)

    @staticmethod
    def _visiting(user: str) -> bool:
        return user.startswith(f"{VISITOR}#")

    def _allowance(self, user: str) -> tuple[str, int, str, int, str]:
        if self._visiting(user):
            return user, self.per_visitor, f"visitors#{self.day()}", self.visitors_daily, "Sign in for more."
        return f"user#{user}", self.per_user, f"day#{self.day()}", self.daily, "Try again tomorrow."

    def _give_back(self, user: str, expires: int, day: bool) -> None:
        mine, each, pool, total, _ = self._allowance(user)
        self.table.add(mine, f"evals#{self.day()}", -1, each + 1, expires)
        if day:
            self.table.add(EVAL, pool, -1, total + 1, expires)

    def request(self, user: str, riot_id: str, again: bool = False):
        riot_id = riot_id_of(riot_id)
        if self.store_size() >= self.cap_bytes:
            raise LinkError("Our game store is full right now. Try again tomorrow.")
        mine, each, pool, total, after = self._allowance(user)
        expires = int(self.clock()) + 2 * DAY
        if self.table.add(mine, f"evals#{self.day()}", 1, each, expires) is None:
            raise QuotaExceeded(f"You can ask for {each} new players a day. {after}")
        if self.table.add(EVAL, pool, 1, total, expires) is None:
            self._give_back(user, expires, day=False)
            raise LinkError(f"We've pulled as many new players as we can today. {after}")
        try:
            name = self._start(user, riot_id, again)
        except Exception:
            self._give_back(user, expires, day=True)
            raise
        if again:
            raise LinkError(f"We're pulling {name}'s newest games now. Check back in about 10 minutes.")
        raise LinkError(f"We don't know {name} yet. We're pulling their last {GAMES} ranked games now. Check back in about 10 minutes.")

    def _start(self, user: str, riot_id: str, again: bool) -> str:
        budget = self.visitor_budget if self._visiting(user) else self.budget
        if not budget(1):
            raise LinkError("Riot is busy right now. Try again in a few minutes.")
        account = self.riot.account(riot_id)
        puuid = account["puuid"]
        name = f"{account.get('gameName', riot_id.partition('#')[0])}#{account.get('tagLine', riot_id.partition('#')[2])}"
        now = int(self.clock())
        self.table.put(NAMES, name_key(name), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        if name_key(name) != name_key(riot_id):
            self.table.put(NAMES, name_key(riot_id), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        self._held(self.status(puuid), name, again)
        state = {"status": REQUESTED, "riot_id": name, "started": now, "expires": now + self.keep_days * DAY, "by": user}
        self.table.put(EVAL, puuid, state)
        if self.launch is not None:
            try:
                self.launch(puuid)
            except Exception:
                self.table.put(EVAL, puuid, {**state, "status": FAILED})
                raise LinkError(f"We couldn't start pulling {name}'s games. Try again later.")
        return name
