import math
from itertools import combinations

from .regions import MAP_SPAN, REGIONS, region_of
from .timeline import ParsedTimeline

SPAN = 15
LINK_RANGE = 2000.0
BREAK_RANGE = 5000.0
BANDS = (1500.0, 3000.0, 5000.0)
DEATH_WINDOW = 0.5
AVENGE_WINDOW = 1.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    dx = [value - mx for value in xs[:n]]
    dy = [value - my for value in ys[:n]]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0.0:
        return 0.0
    return sum((index - mean_x) * (values[index] - mean_y) for index in range(n)) / denominator


def _lead_lag(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    toward_b = sum(
        1
        for index in range(1, n)
        if math.dist(a[index], b[index - 1]) < math.dist(a[index - 1], b[index - 1])
    )
    toward_a = sum(
        1
        for index in range(1, n)
        if math.dist(b[index], a[index - 1]) < math.dist(b[index - 1], a[index - 1])
    )
    total = toward_a + toward_b
    return (toward_b - toward_a) / total if total else 0.0


def dyad_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    kills = [
        event
        for event in parsed.kill_events()
        if float(event.get("timestamp", 0)) / 60000.0 <= limit
    ]
    zones = {
        pid: [
            REGIONS[region_of(x, y, parsed.teams.get(pid, 100))]
            for x, y in (parsed.positions.get(pid) or [])[: limit + 1]
        ]
        for pid in parsed.pid_to_puuid
    }
    rows = []
    for team in (100, 200):
        members = sorted(
            pid for pid, value in parsed.teams.items() if value == team and pid in parsed.pid_to_puuid
        )
        win = int(bool(next(
            p.get("win") for p in match["info"]["participants"] if int(p.get("teamId", 0)) == team
        )))
        for left, right in combinations(members, 2):
            track_a = (parsed.positions.get(left) or [])[: limit + 1]
            track_b = (parsed.positions.get(right) or [])[: limit + 1]
            frames = min(len(track_a), len(track_b))
            if frames < 4:
                continue
            distances = [
                math.dist(track_a[i], track_b[i])
                for i in range(1, frames)
                if parsed.alive_at(left, float(i)) and parsed.alive_at(right, float(i))
            ]
            if not distances:
                continue
            links, linked = 0, distances[0] < LINK_RANGE
            for value in distances[1:]:
                if not linked and value < LINK_RANGE:
                    links += 1
                    linked = True
                elif linked and value > BREAK_RANGE:
                    linked = False
            takedowns_a, takedowns_b, shared = set(), set(), 0
            deaths_a, deaths_b = [], []
            avenged = 0
            takedown_times_a, takedown_times_b = [], []
            for index, event in enumerate(kills):
                involved = parsed.involved(event)
                minute = float(event.get("timestamp", 0)) / 60000.0
                if left in involved:
                    takedowns_a.add(index)
                    takedown_times_a.append(minute)
                if right in involved:
                    takedowns_b.add(index)
                    takedown_times_b.append(minute)
                if left in involved and right in involved:
                    shared += 1
                victim = int(event.get("victimId", 0))
                if victim == left:
                    deaths_a.append(minute)
                if victim == right:
                    deaths_b.append(minute)
            union = len(takedowns_a | takedowns_b)
            co_deaths = sum(1 for a in deaths_a if any(abs(a - b) <= DEATH_WINDOW for b in deaths_b))
            avenged = sum(
                1 for when in deaths_a if any(0 < t - when <= AVENGE_WINDOW for t in takedown_times_b)
            ) + sum(
                1 for when in deaths_b if any(0 < t - when <= AVENGE_WINDOW for t in takedown_times_a)
            )
            seen_a = set(zones.get(left, [])[:frames])
            seen_b = set(zones.get(right, [])[:frames])
            overlap = len(seen_a & seen_b) / len(seen_a | seen_b) if (seen_a | seen_b) else 0.0
            gold_a = (parsed.gold.get(left) or [])[: limit + 1]
            gold_b = (parsed.gold.get(right) or [])[: limit + 1]
            damage_a = (parsed.damage_done.get(left) or [])[: limit + 1]
            damage_b = (parsed.damage_done.get(right) or [])[: limit + 1]
            both_fighting = sum(
                1
                for index in range(1, min(len(damage_a), len(damage_b)))
                if damage_a[index] > damage_a[index - 1] and damage_b[index] > damage_b[index - 1]
            )
            either_fighting = sum(
                1
                for index in range(1, min(len(damage_a), len(damage_b)))
                if damage_a[index] > damage_a[index - 1] or damage_b[index] > damage_b[index - 1]
            )
            row = {
                "match_id": parsed.match_id,
                "puuid_a": parsed.pid_to_puuid[left],
                "puuid_b": parsed.pid_to_puuid[right],
                "team_id": team,
                "role_a": parsed.roles.get(left, "") or "UNKNOWN",
                "role_b": parsed.roles.get(right, "") or "UNKNOWN",
                "win": win,
                "d_mean_distance": sum(distances) / len(distances) / MAP_SPAN,
                "d_min_distance": min(distances) / MAP_SPAN,
                "d_distance_trend": _slope(distances) / MAP_SPAN,
                "d_links": float(links),
                "d_link_rate": links / max(frames - 1, 1),
                "d_lead_lag": _lead_lag(track_a, track_b),
                "d_co_takedown_rate": shared / union if union else 0.0,
                "d_takedown_union": float(union),
                "d_co_death_rate": co_deaths / max(min(len(deaths_a), len(deaths_b)), 1),
                "d_avenged_rate": avenged / max(len(deaths_a) + len(deaths_b), 1),
                "d_zone_overlap": overlap,
                "d_gold_corr": _pearson(
                    [gold_a[i + 1] - gold_a[i] for i in range(len(gold_a) - 1)],
                    [gold_b[i + 1] - gold_b[i] for i in range(len(gold_b) - 1)],
                ),
                "d_fight_together": both_fighting / either_fighting if either_fighting else 0.0,
            }
            for band in BANDS:
                row[f"d_close_share_{int(band)}"] = sum(1 for value in distances if value < band) / len(distances)
            rows.append(row)
    return rows


DYAD_FEATURE_COLUMNS = [
    "d_mean_distance",
    "d_min_distance",
    "d_distance_trend",
    "d_links",
    "d_link_rate",
    "d_lead_lag",
    "d_co_takedown_rate",
    "d_takedown_union",
    "d_co_death_rate",
    "d_avenged_rate",
    "d_zone_overlap",
    "d_gold_corr",
    "d_fight_together",
    *[f"d_close_share_{int(band)}" for band in BANDS],
]
