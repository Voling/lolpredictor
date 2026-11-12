import logging

import httpx
import pytest

from synergy.api.accounts import LinkError, LocalTable, NotFound, RateLimited, RiotAccounts, RiotRefused, RiotUnavailable
from synergy.api.limiter import LIMITS, RiotLimiter, patiently, windows_of

START = 1_790_000_040.0
PERSONAL = "20/1,100/120"


def test_limits_are_read_longest_window_first_and_filled_to_ninety_percent():
    # given
    texts = (PERSONAL, " 500/10 , 30000/600 ")

    # when
    windows = [windows_of(text) for text in texts]

    # then
    assert windows == [((90, 120), (18, 1)), ((27000, 600), (450, 10))]
    with pytest.raises(ValueError):
        windows_of(" ")


def test_each_window_starts_at_its_first_call_and_refuses_past_its_share_until_it_ends():
    # given
    first = START + 60.25
    now = {"t": first}
    limiter = RiotLimiter(LocalTable(), PERSONAL, clock=lambda: now["t"])

    # when
    burst = [limiter.take("americas") for _ in range(19)]
    for second in range(1, 5):
        now["t"] = first + second
        burst += [limiter.take("americas") for _ in range(18)]
    now["t"] = START + 120
    across_the_minute = limiter.take("americas")
    other_route = limiter.take("na1")
    now["t"] = first + 120
    reopened = limiter.take("americas")

    # then
    assert burst[:18] == [0.0] * 18 and burst[18] == pytest.approx(1.0)
    assert burst[19:] == [0.0] * 72 and across_the_minute == pytest.approx(60.25)
    assert other_route == 0.0 and reopened == 0.0


def test_a_refusal_by_a_later_window_gives_the_calls_back_to_the_earlier_ones():
    # given
    table = LocalTable()
    limiter = RiotLimiter(table, PERSONAL, clock=lambda: START)
    for _ in range(9):
        limiter.take("americas", 2)

    # when
    refused = limiter.take("americas", 2)

    # then
    assert refused == pytest.approx(1.0) and table.get(LIMITS, "americas#120")["used"] == 18 and table.get(LIMITS, "americas#1")["used"] == 18


def test_two_callers_racing_past_the_end_of_a_window_start_only_one_new_window():
    # given
    now = {"t": START}
    table = LocalTable()
    one, two = RiotLimiter(table, PERSONAL, clock=lambda: now["t"]), RiotLimiter(table, PERSONAL, clock=lambda: now["t"])
    one.take("americas")
    now["t"] = START + 130
    ended = table.get(LIMITS, "americas#120")["start"]

    # when
    won = one.take("americas")
    lost = table.start_window(LIMITS, "americas#120", 1, START + 130.5, ended, 0)
    joined = two.take("americas")

    # then
    window = table.get(LIMITS, "americas#120")
    assert won == 0.0 and lost is False and joined == 0.0 and window["start"] == START + 130 and window["used"] == 2


def test_a_cooldown_holds_every_caller_on_its_route_until_it_ends_and_never_shortens():
    # given
    now = {"t": START}
    table = LocalTable()
    one, two = RiotLimiter(table, PERSONAL, clock=lambda: now["t"]), RiotLimiter(table, PERSONAL, clock=lambda: now["t"])

    # when
    one.cool("americas", 30)
    one.cool("americas", 5)
    held = two.take("americas")
    platform = two.take("na1")
    now["t"] += 30
    after = two.take("americas")

    # then
    assert held == pytest.approx(30.0) and platform == 0.0 and after == 0.0


def test_a_429_cools_its_route_for_retry_after_and_each_call_charges_its_own_route():
    # given
    table = LocalTable()

    def answer(request):
        if "/league/" in request.url.path:
            return httpx.Response(200, json=[])
        if "/matches/" in request.url.path:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"puuid": "p"})

    riot = RiotAccounts(lambda: "key", "na1", "americas", RiotLimiter(table, PERSONAL, clock=lambda: START), httpx.Client(transport=httpx.MockTransport(answer)))

    # when
    riot.account("a#na1")
    with pytest.raises(RateLimited) as limited:
        riot.game("NA1_1")
    with pytest.raises(RateLimited) as held:
        riot.account("a#na1")
    leagues = [riot.league("p"), riot.league("p")]

    # then
    assert limited.value.retry_after == 7.0 and held.value.retry_after == pytest.approx(7.0) and leagues == [[], []]
    assert table.get(LIMITS, "americas#120")["used"] == 3 and table.get(LIMITS, "na1#120")["used"] == 2


def test_patience_waits_out_refusals_then_gives_up_past_its_limit():
    # given
    answers = [RateLimited(7.0), RateLimited(0.0), "ok"]
    naps, now = [], {"t": 0.0}

    def call():
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def always_busy():
        raise RateLimited(400.0)

    def nap(seconds):
        naps.append(seconds)
        now["t"] += seconds

    # when
    found = patiently(call, nap, lambda: now["t"])
    waited = list(naps)
    with pytest.raises(RateLimited):
        patiently(always_busy, nap, lambda: now["t"], patience=600.0)

    # then
    assert found == "ok" and waited == [7.0, 0.001] and naps[2:] == [120.0, 120.0]


def test_a_rejected_key_trips_one_breaker_for_every_caller_logs_once_per_trip_and_clears_when_it_ends(caplog):
    # given
    now, asked, keys = {"t": START}, [], {"value": "dead-key"}
    table = LocalTable()

    def answer(request):
        asked.append(request.headers["X-Riot-Token"])
        return httpx.Response(403 if request.headers["X-Riot-Token"] == "dead-key" else 200, json={"puuid": "p"})

    def key():
        return keys["value"]

    def riot():
        return RiotAccounts(key, "na1", "americas", RiotLimiter(table, PERSONAL, clock=lambda: now["t"], breaker_seconds=600), httpx.Client(transport=httpx.MockTransport(answer)))

    one, two = riot(), riot()

    # when
    with caplog.at_level(logging.ERROR, logger="synergy.api.limiter"):
        with pytest.raises(RiotUnavailable) as tripped:
            one.account("a#na1")
        now["t"] += 100
        with pytest.raises(RiotUnavailable) as held:
            two.league("p")
        now["t"] += 500
        with pytest.raises(RiotUnavailable) as again:
            two.account("a#na1")
    keys["value"] = "new-key"
    rotated = one.account("a#na1")

    # then
    assert (tripped.value.retry_after, held.value.retry_after, again.value.retry_after) == (600.0, pytest.approx(500.0), 600.0)
    assert asked == ["dead-key", "dead-key", "new-key"] and rotated == {"puuid": "p"} and str(held.value) == "Riot isn't answering us right now. Try again later."
    assert [record.getMessage() for record in caplog.records] == ["riot key rejected with 403, holding every Riot call for 600s"] * 2


def test_patience_never_waits_out_a_rejected_key():
    # given
    naps = []

    def rejected():
        raise RiotUnavailable(30.0)

    # when
    with pytest.raises(RiotUnavailable):
        patiently(rejected, naps.append, lambda: 0.0)

    # then
    assert naps == []


def test_riot_answers_are_sorted_into_permanent_and_passing_failures():
    # given
    codes = {"NA1_1": 404, "NA1_2": 400, "NA1_3": 503, "NA1_4": 500}

    def answer(request):
        return httpx.Response(codes[request.url.path.rsplit("/", 1)[-1]])

    riot = RiotAccounts(lambda: "key", "na1", "americas", RiotLimiter(LocalTable(), PERSONAL, clock=lambda: START), httpx.Client(transport=httpx.MockTransport(answer)))

    # when
    found = {}
    for match_id in codes:
        try:
            riot.match(match_id)
        except LinkError as error:
            found[match_id] = error

    # then
    assert isinstance(found["NA1_1"], NotFound) and type(found["NA1_2"]) is RiotRefused
    assert all(type(found[match_id]) is LinkError for match_id in ("NA1_3", "NA1_4"))
