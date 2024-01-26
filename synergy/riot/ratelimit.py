import asyncio
from collections import deque


class SlidingWindow:
    def __init__(self, limit: int, window: float, clock=None):
        self.limit = limit
        self.window = window
        self.clock = clock or asyncio.get_event_loop().time
        self._hits: deque[float] = deque()

    def delay(self) -> float:
        now = self.clock()
        while self._hits and now - self._hits[0] >= self.window:
            self._hits.popleft()
        if len(self._hits) < self.limit:
            return 0.0
        return self.window - (now - self._hits[0])

    def record(self) -> None:
        self._hits.append(self.clock())


class RateLimiter:
    def __init__(self, windows: list[tuple[int, float]], clock=None):
        self._clock = clock
        self._windows: list[SlidingWindow] | None = None
        self._spec = windows
        self._lock: asyncio.Lock | None = None
        self._penalty_until = 0.0

    def _init(self) -> None:
        if self._windows is None:
            clock = self._clock or asyncio.get_event_loop().time
            self._windows = [SlidingWindow(limit, window, clock) for limit, window in self._spec]
            self._lock = asyncio.Lock()

    def _now(self) -> float:
        return (self._clock or asyncio.get_event_loop().time)()

    async def acquire(self) -> None:
        self._init()
        assert self._lock is not None and self._windows is not None
        while True:
            async with self._lock:
                wait = max([w.delay() for w in self._windows] + [self._penalty_until - self._now()])
                if wait <= 0:
                    for w in self._windows:
                        w.record()
                    return
            await asyncio.sleep(wait)

    def penalise(self, seconds: float) -> None:
        self._init()
        self._penalty_until = max(self._penalty_until, self._now() + seconds)
