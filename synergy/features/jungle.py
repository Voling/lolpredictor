import math

from .regions import REGIONS, region_of
from .timeline import ParsedTimeline

GANK_RANGE = 2200.0
GANK_FIRST_MINUTE = 2
OPENING_MINUTES = 6
LANES = ("LANE_TOP", "LANE_MID", "LANE_BOT")
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
TOPSIDE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_ENEMY_TOPSIDE"}
BOTSIDE = {"JUNGLE_OWN_BOTSIDE", "JUNGLE_ENEMY_BOTSIDE"}


def _zones(parsed: ParsedTimeline, pid: int, team: int, span: int) -> list[str]:
    return [REGIONS[region_of(x, y, team)] for x, y in (parsed.positions.get(pid) or [])[: span + 1]]


def _first(values: list[bool], default: float) -> float:
    return next((index for index, flag in enumerate(values) if flag), default)


def classify(farm: list[float], gank: float, invade: float, span: int) -> str:
    if invade <= 1:
        return "level_one_invade"
    if invade < span and farm[min(int(invade), len(farm) - 1)] < 12:
        return "early_invade"
    if gank >= span:
        return "full_clear"
    farmed = farm[min(int(gank), len(farm) - 1)]
    if farmed < 8:
        return "early_gank"
    if farmed < 20:
        return "three_camp_gank"
    return "late_gank"


def jungle_openings(
    match: dict, timeline: dict, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = min(len(parsed.minutes) - 1, OPENING_MINUTES)
    rows = []
    for team in (100, 200):
        pid = parsed.role_pid(team, "JUNGLE")
        if pid is None:
            continue
        zones = _zones(parsed, pid, team, span)
        if len(zones) <= span:
            continue
        farm = (parsed.jungle_cs.get(pid) or [])[: span + 1]
        if not farm:
            continue
        start = next((zone for zone in zones[1:] if zone in OWN_JUNGLE | ENEMY_JUNGLE), "UNKNOWN")
        invade = _first([zone in ENEMY_JUNGLE for zone in zones], float(span))
        enemies = [
            other
            for other in parsed.pid_to_puuid
            if parsed.teams.get(other) != team and parsed.roles.get(other) != "JUNGLE"
        ]
        track = parsed.positions.get(pid) or []
        ganks = []
        for minute in range(min(len(zones), len(track))):
            if minute < GANK_FIRST_MINUTE or not any(zones[minute].startswith(p) for p in LANES):
                ganks.append(False)
                continue
            ganks.append(
                any(
                    minute < len(parsed.positions.get(other) or [])
                    and math.hypot(
                        track[minute][0] - parsed.positions[other][minute][0],
                        track[minute][1] - parsed.positions[other][minute][1],
                    )
                    < GANK_RANGE
                    for other in enemies
                )
            )
        gank = _first(ganks, float(span))
        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": parsed.pid_to_puuid[pid],
                "team_id": team,
                "champion": next(
                    (
                        p.get("championName")
                        for p in match["info"]["participants"]
                        if p["puuid"] == parsed.pid_to_puuid[pid]
                    ),
                    "",
                ),
                "start_side": "topside" if start in TOPSIDE else "botside" if start in BOTSIDE else "unknown",
                "opening": classify(farm, gank, invade, span),
                "first_invade_minute": invade,
                "first_gank_minute": gank,
                "jungle_cs_at_3": farm[min(3, len(farm) - 1)],
                "jungle_cs_at_5": farm[min(5, len(farm) - 1)],
                "crossed_sides": bool(
                    (zones[1] in TOPSIDE and zones[min(5, len(zones) - 1)] in BOTSIDE)
                    or (zones[1] in BOTSIDE and zones[min(5, len(zones) - 1)] in TOPSIDE)
                ),
                "win": int(bool(next(
                    p.get("win") for p in match["info"]["participants"] if int(p.get("teamId", 0)) == team
                ))),
            }
        )
    return rows


JUNGLE_OPENINGS = (
    "level_one_invade",
    "early_invade",
    "early_gank",
    "three_camp_gank",
    "late_gank",
    "full_clear",
)
