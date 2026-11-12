import time
from datetime import datetime, timezone

from ..ml.evaluated import FRESH_SECONDS
from .accounts import DAY, RIOT_BUSY, VISITOR, LinkError, QuotaExceeded, RateLimited, RiotBusy, RiotUnavailable, riot_id_of

EVAL = "eval"
NAMES = "evalname"
STORE = "evaluated"
JOBS = "history"
REQUESTED, RUNNING, READY, FAILED = "requested", "running", "ready", "failed"
PLAYER, SYSTEM = "player", "system"
UNDER_WAY = (REQUESTED, RUNNING, READY)
GAMES = 50
STALE_SECONDS = 1800
KEEP_DAYS = 14
GIB = 1024**3
SYSTEM_PROBLEM = "We hit a problem pulling {name}'s games. Try again."
NOT_STARTED = "We couldn't start pulling {name}'s games. Try again later."
RUN_FIELDS = ("pk", "sk", "task", "error", "kind")
WEIGHTS = {"fetching": (0, 50), "loading": (50, 60), "features": (60, 70), "positions": (70, 78), "reactions": (78, 82), "tendencies": (82, 86), "habits": (86, 88), "movement": (88, 92), "embedding": (92, 97), "scoring": (97, 100)}


class Superseded(Exception):
    pass


class Refunded(LinkError):
    pass


class Progress:
    def __init__(self, table, puuid: str, clock=time.time, owner: str | None = None):
        self.table, self.puuid, self.clock, self.owner = table, puuid, clock, owner
        self.state = table.get(EVAL, puuid) or {}

    def write(self, **changes) -> None:
        self.state = {**self.state, **changes, "updated": int(self.clock())}
        if self.owner is None:
            self.table.put(EVAL, self.puuid, self.state)
        elif not self.table.put_if(EVAL, self.puuid, self.state, {"task": self.owner}):
            raise Superseded(self.puuid)

    def step(self, name: str, done: int = 0, total: int = 0) -> None:
        self.write(status=RUNNING, step=name, done=int(done), total=int(total))


def percent_of(state: dict) -> int:
    if state.get("status") == READY:
        return 100
    low, high = WEIGHTS.get(state.get("step"), (0, 0))
    total = int(state.get("total", 0) or 0)
    share = min(1.0, int(state.get("done", 0) or 0) / total) if total else 0.0
    return int(low + (high - low) * share)


def since(state: dict) -> int:
    return int(state.get("updated") or state.get("started") or 0)


def unchanged(state: dict) -> dict:
    return {name: state.get(name) for name in ("status", "updated", "task")}


def queued_state(state: dict, now: int, drop: tuple[str, ...] = (), **changes) -> dict:
    kept = {name: value for name, value in state.items() if name not in RUN_FIELDS and name not in drop}
    return {**kept, "status": REQUESTED, "step": "queued", "updated": now, **changes}


def counters(user: str, day: str) -> tuple[str, str]:
    if user.startswith(f"{VISITOR}#"):
        return user, f"visitors#{day}"
    return f"user#{user}", f"day#{day}"


def day_of(moment: float) -> str:
    return f"{datetime.fromtimestamp(moment, timezone.utc):%Y-%m-%d}"


def give_back(table, puuid: str, state: dict) -> bool:
    user, started = state.get("by"), state.get("started")
    if not user or started is None or not table.raise_to(EVAL, puuid, "refunded", 1, {"by": user, "started": started}):
        return False
    day, expires = day_of(int(started)), int(started) + 2 * DAY
    mine, pool = counters(user, day)
    table.add(mine, f"evals#{day}", -1, 1 << 62, expires)
    table.add(EVAL, pool, -1, 1 << 62, expires)
    return True


def fail_evaluation(table, puuid: str, message: str, kind: str, clock=time.time, match: dict | None = None) -> bool:
    state = table.get(EVAL, puuid) or {}
    now = int(clock())
    kept = {name: value for name, value in state.items() if name not in ("pk", "sk", "task")}
    failed = {"expires": now + KEEP_DAYS * DAY, **kept, "status": FAILED, "step": "failed", "error": message.replace("{name}", str(state.get("riot_id") or "this player")), "kind": kind, "updated": now}
    if match is None:
        table.put(EVAL, puuid, failed)
    elif not table.put_if(EVAL, puuid, failed, match):
        return False
    table.delete(JOBS, puuid)
    if kind == SYSTEM:
        give_back(table, puuid, failed)
    return True


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
        "percent": percent_of(state),
        "message": message or state.get("error"),
    }


class Evaluations:
    def __init__(
        self,
        table,
        riot,
        launch=None,
        store_size=lambda: 0,
        cap_bytes: int = 10 * GIB,
        daily: int = 20,
        per_user: int = 3,
        per_visitor: int = 3,
        visitors_daily: int = 150,
        visitor_budget=None,
        keep_days: int = KEEP_DAYS,
        clock=time.time,
    ):
        self.table, self.riot, self.launch, self.store_size = table, riot, launch, store_size
        self.cap_bytes, self.daily, self.per_user, self.keep_days, self.clock = cap_bytes, daily, per_user, keep_days, clock
        self.per_visitor, self.visitors_daily, self.visitor_budget = per_visitor, visitors_daily, visitor_budget or (lambda calls: True)

    def day(self) -> str:
        return day_of(self.clock())

    def status(self, puuid: str) -> dict | None:
        return self.table.get(EVAL, puuid)

    def lookup(self, riot_id: str) -> dict | None:
        found = self.table.get(NAMES, name_key(riot_id))
        return self.status(found["puuid"]) if found else None

    def _held(self, state: dict | None, riot_id: str, again: bool) -> None:
        if not state:
            return
        now = int(self.clock())
        name, quiet, status = state.get("riot_id", riot_id), now - since(state), state.get("status")
        if status in (REQUESTED, RUNNING) and quiet < STALE_SECONDS:
            raise LinkError(f"We're still pulling {name}'s games. Try again in a few minutes.")
        if again:
            return
        if status == READY and now < int(state.get("expires", 0)):
            raise LinkError(f"We have {name}'s games but the model hasn't picked them up yet. Try again in a minute.")
        if status == FAILED and state.get("kind") != SYSTEM and quiet < DAY:
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
        mine, pool = counters(user, self.day())
        if self._visiting(user):
            return mine, self.per_visitor, pool, self.visitors_daily, "Sign in for more."
        return mine, self.per_user, pool, self.daily, "Try again tomorrow."

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
        except Refunded:
            raise
        except Exception:
            self._give_back(user, expires, day=True)
            raise
        if again:
            raise LinkError(f"We're pulling {name}'s newest games now. Check back in about 10 minutes.")
        raise LinkError(f"We don't know {name} yet. We're pulling their last {GAMES} ranked games now. Check back in about 10 minutes.")

    def _start(self, user: str, riot_id: str, again: bool) -> str:
        visiting = self._visiting(user)
        if visiting and not self.visitor_budget(1):
            raise RiotBusy(RIOT_BUSY)
        try:
            account = self.riot.account(riot_id)
        except RiotUnavailable:
            raise
        except RateLimited as busy:
            if visiting:
                raise RiotBusy(RIOT_BUSY) from busy
            raise
        puuid = account["puuid"]
        name = f"{account.get('gameName', riot_id.partition('#')[0])}#{account.get('tagLine', riot_id.partition('#')[2])}"
        now = int(self.clock())
        self.table.put(NAMES, name_key(name), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        if name_key(name) != name_key(riot_id):
            self.table.put(NAMES, name_key(riot_id), {"puuid": puuid, "expires": now + self.keep_days * DAY})
        self._held(self.status(puuid), name, again)
        state = {"status": REQUESTED, "riot_id": name, "started": now, "updated": now, "expires": now + self.keep_days * DAY, "by": user}
        self.table.put(EVAL, puuid, state)
        if self.launch is not None:
            try:
                self.launch(puuid)
            except Exception:
                fail_evaluation(self.table, puuid, NOT_STARTED, SYSTEM, self.clock)
                raise Refunded(NOT_STARTED.replace("{name}", name))
        return name
