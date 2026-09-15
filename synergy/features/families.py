import math

from .regions import REGIONS, region_of
from .timeline import ParsedTimeline
from .wave import PUSH, wave_states

SPAN = 15
ARRIVE = 2000.0
LANE_ROLES = ("TOP", "MIDDLE", "BOTTOM", "UTILITY")
LANE_PREFIX = {"TOP": "LANE_TOP", "MIDDLE": "LANE_MID", "BOTTOM": "LANE_BOT", "UTILITY": "LANE_BOT"}

FAMILIES = {
    frozenset({"BOTTOM", "UTILITY"}): "lane_partners",
    frozenset({"JUNGLE", "TOP"}): "jungler_laner",
    frozenset({"JUNGLE", "MIDDLE"}): "jungler_laner",
    frozenset({"JUNGLE", "BOTTOM"}): "jungler_laner",
    frozenset({"JUNGLE", "UTILITY"}): "jungler_laner",
    frozenset({"TOP", "MIDDLE"}): "cross_map",
    frozenset({"TOP", "BOTTOM"}): "cross_map",
    frozenset({"MIDDLE", "BOTTOM"}): "cross_map",
    frozenset({"TOP", "UTILITY"}): "roaming_support",
    frozenset({"MIDDLE", "UTILITY"}): "roaming_support",
}


def family_of(left: str, right: str) -> str:
    return FAMILIES.get(frozenset({left, right}), "other")


def _side(x: float, y: float) -> str:
    return "top" if y > x else "bottom"


def family_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    zones = {
        pid: [
            REGIONS[region_of(x, y, parsed.teams.get(pid, 100))]
            for x, y in (parsed.positions.get(pid) or [])[: limit + 1]
        ]
        for pid in parsed.pid_to_puuid
    }
    waves = {pid: wave_states(parsed, pid, limit) for pid in parsed.pid_to_puuid}
    deaths: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    takedowns: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    for event in parsed.kill_events():
        minute = int(float(event.get("timestamp", 0)) // 60000)
        if minute > limit:
            continue
        victim = int(event.get("victimId", 0))
        if victim in deaths:
            deaths[victim].add(minute)
        for pid in parsed.involved(event):
            if pid in takedowns:
                takedowns[pid].add(minute)
    rows = []
    for team in (100, 200):
        members = sorted(pid for pid in parsed.pid_to_puuid if parsed.teams.get(pid) == team)
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                role_a, role_b = parsed.roles.get(left, ""), parsed.roles.get(right, "")
                family = family_of(role_a, role_b)
                if family == "other":
                    continue
                jungler, laner = (left, right) if role_a == "JUNGLE" else (right, left)
                track_a = parsed.positions.get(left) or []
                track_b = parsed.positions.get(right) or []
                for minute in range(1, min(limit, len(track_a), len(track_b))):
                    row = {
                        "match_id": parsed.match_id,
                        "pair_key": "|".join(sorted((parsed.pid_to_puuid[left], parsed.pid_to_puuid[right]))),
                        "puuid_a": parsed.pid_to_puuid[left],
                        "puuid_b": parsed.pid_to_puuid[right],
                        "role_pair": "-".join(sorted((role_a, role_b))),
                        "family": family,
                        "minute": minute,
                        "opportunity": 0.0,
                        "converted": 0.0,
                    }
                    if family == "jungler_laner":
                        home = LANE_PREFIX.get(parsed.roles.get(laner, ""), "")
                        near = math.dist(
                            (parsed.positions[jungler])[minute], (parsed.positions[laner])[minute]
                        ) < ARRIVE
                        in_lane = home and zones[laner][minute].startswith(home)
                        row["opportunity"] = float(near and in_lane)
                        if row["opportunity"]:
                            window = {minute, minute + 1}
                            row["converted"] = float(
                                bool(takedowns[jungler] & window) and bool(takedowns[laner] & window)
                            )
                    elif family == "lane_partners":
                        together = math.dist(track_a[minute], track_b[minute]) < ARRIVE
                        row["opportunity"] = float(together)
                        if together:
                            window = {minute, minute + 1}
                            row["converted"] = float(
                                bool(takedowns[left] & window) and bool(takedowns[right] & window)
                            )
                    elif family == "cross_map":
                        pushing_a = minute < len(waves[left]) and waves[left][minute] == PUSH
                        pushing_b = minute < len(waves[right]) and waves[right][minute] == PUSH
                        row["opportunity"] = float(pushing_a or pushing_b)
                        if row["opportunity"]:
                            row["converted"] = float(
                                _side(*track_a[minute]) != _side(*track_b[minute])
                            )
                    else:
                        support, other = (left, right) if role_a == "UTILITY" else (right, left)
                        home = LANE_PREFIX.get(parsed.roles.get(other, ""), "")
                        arrived = math.dist(
                            (parsed.positions[support])[minute], (parsed.positions[other])[minute]
                        ) < ARRIVE
                        away_from_bot = not zones[support][minute].startswith("LANE_BOT")
                        row["opportunity"] = float(arrived and away_from_bot)
                        if row["opportunity"]:
                            window = {minute, minute + 1}
                            row["converted"] = float(bool(takedowns[other] & window))
                    if row["opportunity"]:
                        rows.append(row)
    return rows
