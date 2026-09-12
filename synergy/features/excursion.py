import math

from .anchors import MAX_REACH, confidence_at
from .regions import REGIONS, region_of
from .timeline import ParsedTimeline

SPAN = 15
INTERVENE = 2000.0
VISIT_GAP = 0.35


def _region(x: float, y: float, team: int) -> str:
    return REGIONS[region_of(x, y, team)]


def excursion_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    deaths = []
    for event in parsed.kill_events():
        position = event.get("position") or {}
        victim = int(event.get("victimId", 0))
        if victim and "x" in position:
            deaths.append((float(event.get("timestamp", 0)) / 60000.0, victim,
                           float(position["x"]), float(position["y"])))
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        team = parsed.teams.get(pid, 100)
        track = parsed.positions.get(pid) or []
        anchors = [a for a in parsed.anchors.get(pid, ()) if a[0] <= limit]
        hidden = returned = jungle = visits = enemy_half = 0
        reaches = []
        last_shop = None
        for when, x, y, _ in anchors:
            low, high = int(when), int(when) + 1
            if low < 1 or high >= len(track):
                continue
            here = _region(x, y, team)
            before = _region(track[low][0], track[low][1], team)
            after = _region(track[high][0], track[high][1], team)
            if here == "BASE_OWN":
                if last_shop is None or when - last_shop > VISIT_GAP:
                    visits += 1
                last_shop = when
            if here.endswith("ENEMY") or here.startswith("JUNGLE_ENEMY"):
                enemy_half += 1
            if here != before and here != after:
                hidden += 1
                reaches.append(max(math.hypot(track[low][0] - x, track[low][1] - y),
                                   math.hypot(track[high][0] - x, track[high][1] - y)))
                if before == after:
                    returned += 1
                if here.startswith("JUNGLE") and before.startswith("LANE"):
                    jungle += 1
        interventions = 0
        for when, victim, x, y in deaths:
            if when > limit or parsed.teams.get(victim) != team or victim == pid:
                continue
            if not parsed.alive_at(pid, when):
                continue
            spot, _ = confidence_at(anchors, track, when)
            if math.hypot(spot[0] - x, spot[1] - y) < INTERVENE:
                interventions += 1
        span_minutes = max(limit, 1)
        radii = [confidence_at(anchors, track, m + 0.5)[1] for m in range(limit)]
        rows.append({
            "match_id": parsed.match_id,
            "puuid": puuid,
            "x_anchors_pm": len(anchors) / span_minutes,
            "x_hidden_share": hidden / len(anchors) if anchors else 0.0,
            "x_return_share": returned / hidden if hidden else 0.0,
            "x_jungle_excursions": float(jungle),
            "x_base_visits": float(visits),
            "x_enemy_half_anchors": float(enemy_half),
            "x_hidden_reach": (sum(reaches) / len(reaches)) / 1000.0 if reaches else 0.0,
            "x_ally_death_presence": interventions / span_minutes,
            "x_position_doubt": (sum(radii) / len(radii)) / MAX_REACH if radii else 1.0,
        })
    return rows


EXCURSION_COLUMNS = [
    "x_anchors_pm",
    "x_hidden_share",
    "x_return_share",
    "x_jungle_excursions",
    "x_base_visits",
    "x_enemy_half_anchors",
    "x_hidden_reach",
    "x_ally_death_presence",
    "x_position_doubt",
]
