import hashlib
import logging
import time

from .accounts import DAY, RateLimited, RiotAccounts, RiotUnavailable, ssm_key

logger = logging.getLogger(__name__)
LIMITS = "riot#limits"
BREAKER_SECONDS = 600.0
FILL_PERCENT = 90
SOONEST = 0.001
PATIENCE = 600.0
LONGEST_NAP = 120.0
RACES = 5


def windows_of(limits: str) -> tuple[tuple[int, int], ...]:
    windows = []
    for part in str(limits).split(","):
        calls, _, seconds = part.strip().partition("/")
        if calls:
            windows.append((max(1, int(calls) * FILL_PERCENT // 100), int(seconds)))
    if not windows:
        raise ValueError(f"RIOT_LIMITS needs windows like 20/1,100/120, not {limits!r}")
    return tuple(sorted(windows, key=lambda window: -window[1]))


class RiotLimiter:
    def __init__(self, table, limits: str, clock=time.time, breaker_seconds: float = BREAKER_SECONDS):
        self.table, self.windows, self.clock, self.breaker_seconds = table, windows_of(limits), clock, float(breaker_seconds)

    @staticmethod
    def _breaker(key: str) -> str:
        return f"key#{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"

    def held(self, key: str) -> float:
        found = self.table.get(LIMITS, self._breaker(key))
        left = float(found.get("until", 0)) - self.clock() if found else 0.0
        return left if left > 0 else 0.0

    def trip(self, key: str, status: int) -> float:
        if not self.held(key):
            logger.error("riot key rejected with %s, holding every Riot call for %.0fs", status, self.breaker_seconds)
        until = self.clock() + self.breaker_seconds
        self.table.raise_to(LIMITS, self._breaker(key), "until", until, expires=int(until) + DAY)
        return self.breaker_seconds

    def take(self, route: str, calls: int = 1) -> float:
        now = self.clock()
        cooling = self.table.get(LIMITS, f"{route}#cooldown")
        if cooling and float(cooling.get("until", 0)) > now:
            return float(cooling["until"]) - now
        taken = []
        for cap, seconds in self.windows:
            key = f"{route}#{seconds}"
            wait, start = self._charge(key, cap, seconds, calls, now)
            if wait:
                for done, done_cap, done_start in taken:
                    self.table.add_to_window(LIMITS, done, -calls, done_cap + calls, done_start)
                return wait
            taken.append((key, cap, start))
        return 0.0

    def _charge(self, key: str, cap: int, seconds: int, calls: int, now: float) -> tuple[float, float]:
        for _ in range(RACES):
            found = self.table.get(LIMITS, key) or {}
            start = found.get("start")
            if start is not None and now < float(start) + seconds:
                if int(found.get("used", 0)) + calls > cap:
                    return max(float(start) + seconds - now, SOONEST), float(start)
                if self.table.add_to_window(LIMITS, key, calls, cap, float(start)):
                    return 0.0, float(start)
            elif self.table.start_window(LIMITS, key, calls, now, start, int(now + seconds) + DAY):
                return 0.0, now
        return SOONEST, now

    def cool(self, route: str, seconds: float) -> None:
        until = self.clock() + max(float(seconds), 0.0)
        self.table.raise_to(LIMITS, f"{route}#cooldown", "until", until, expires=int(until) + DAY)


def riot_client(settings, table, client=None) -> RiotAccounts:
    key = ssm_key(settings.riot_key_parameter) if settings.riot_key_parameter else (lambda: settings.riot_api_key)
    return RiotAccounts(key, settings.platform, settings.region, RiotLimiter(table, settings.riot_limits, breaker_seconds=settings.riot_breaker_seconds), client)


def patiently(call, pause=time.sleep, clock=time.monotonic, patience: float = PATIENCE):
    started = clock()
    while True:
        try:
            return call()
        except RiotUnavailable:
            raise
        except RateLimited as limited:
            if clock() - started + limited.retry_after > patience:
                raise
            pause(min(max(limited.retry_after, SOONEST), LONGEST_NAP))
