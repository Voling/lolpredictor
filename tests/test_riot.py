import asyncio

import pytest

from synergy.riot.ratelimit import RateLimiter, SlidingWindow
from synergy.riot.routing import UnknownPlatform, join_riot_id, region_for, split_riot_id


def test_region_routing():
    assert region_for("NA1") == "americas"
    assert region_for("euw1") == "europe"
    with pytest.raises(UnknownPlatform):
        region_for("mars1")


def test_riot_id_round_trip():
    assert split_riot_id("bblskibs#gotg") == ("bblskibs", "gotg")
    assert join_riot_id(*split_riot_id(" name # tag ")) == "name#tag"


@pytest.mark.parametrize("value", ["bblskibs", "#gotg", "name#"])
def test_riot_id_rejects_malformed(value):
    with pytest.raises(ValueError):
        split_riot_id(value)


def test_sliding_window_delays_once_the_limit_is_reached():
    now = [0.0]
    window = SlidingWindow(2, 10.0, clock=lambda: now[0])
    assert window.delay() == 0.0
    window.record()
    window.record()
    assert window.delay() == pytest.approx(10.0)
    now[0] = 10.5
    assert window.delay() == 0.0


def test_rate_limiter_spaces_requests_across_windows():
    now = [0.0]
    limiter = RateLimiter([(2, 1.0)], clock=lambda: now[0])
    slept = []

    async def fake_sleep(duration):
        slept.append(duration)
        now[0] += duration

    async def run():
        original = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            for _ in range(3):
                await limiter.acquire()
        finally:
            asyncio.sleep = original

    asyncio.run(run())
    assert slept and slept[0] == pytest.approx(1.0)


def test_rate_limiter_honours_a_penalty():
    now = [0.0]
    limiter = RateLimiter([(10, 1.0)], clock=lambda: now[0])
    slept = []

    async def fake_sleep(duration):
        slept.append(duration)
        now[0] += duration

    async def run():
        original = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            limiter.penalise(5.0)
            await limiter.acquire()
        finally:
            asyncio.sleep = original

    asyncio.run(run())
    assert slept == [pytest.approx(5.0)]
