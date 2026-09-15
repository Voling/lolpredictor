import math

from .timeline import ParsedTimeline
from .wave import lane_progress

SPAN = 15
THREAT_RANGE = 1600.0
PRESSURE_RANGE = 2600.0
RETREAT_PROGRESS = 0.06
RETREAT_DISTANCE = 900.0
LANE_ROLES = ("TOP", "MIDDLE", "BOTTOM", "UTILITY")


def _own_base(team: int) -> tuple[float, float]:
    return (1000.0, 1000.0) if team == 100 else (14000.0, 14000.0)


def _retreated(track, minute: int, team: int) -> tuple[float, float]:
    if minute + 1 >= len(track):
        return 0.0, 0.0
    here, later = track[minute], track[minute + 1]
    drop = lane_progress(*here, team) - lane_progress(*later, team)
    base = _own_base(team)
    homeward = math.dist(here, base) - math.dist(later, base)
    return drop, homeward


def detection_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    deaths: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    for event in parsed.kill_events():
        minute = int(float(event.get("timestamp", 0)) // 60000)
        victim = int(event.get("victimId", 0))
        if victim in deaths and minute <= limit:
            deaths[victim].add(minute)
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        role = parsed.roles.get(pid, "") or "UNKNOWN"
        if role not in LANE_ROLES:
            continue
        team = parsed.teams.get(pid, 100)
        track = parsed.positions.get(pid) or []
        enemies = [other for other in parsed.pid_to_puuid if parsed.teams.get(other) != team]
        for minute in range(1, min(limit, len(track) - 1)):
            if minute in deaths[pid] or (minute + 1) in deaths[pid]:
                continue
            closest, closest_role = float("inf"), ""
            for other in enemies:
                other_track = parsed.positions.get(other) or []
                if minute >= len(other_track):
                    continue
                gap = math.dist(track[minute], other_track[minute])
                if gap < closest:
                    closest, closest_role = gap, parsed.roles.get(other, "") or "UNKNOWN"
            drop, homeward = _retreated(track, minute, team)
            reacted = float(drop > RETREAT_PROGRESS or homeward > RETREAT_DISTANCE)
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "puuid": puuid,
                    "role": role,
                    "team_id": team,
                    "minute": minute,
                    "nearest_enemy": min(closest, 16000.0) / 1000.0,
                    "nearest_role": closest_role,
                    "threat": float(closest < THREAT_RANGE),
                    "pressure": float(closest < PRESSURE_RANGE),
                    "jungler_threat": float(closest < PRESSURE_RANGE and closest_role == "JUNGLE"),
                    "progress_drop": drop,
                    "homeward": homeward / 1000.0,
                    "reacted": reacted,
                    "died_next": float((minute + 1) in deaths[pid] or (minute + 2) in deaths[pid]),
                }
            )
    return rows


def awareness(frame, key: str = "puuid", min_chances: int = 12):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby([key, frame["pressure"] > 0])["reacted"].agg(["size", "mean"])
    table = grouped.unstack(level=1)
    table.columns = ["quiet_n", "threat_n", "false_alarm", "hit"]
    table = table.dropna()
    table = table[(table.threat_n >= min_chances) & (table.quiet_n >= min_chances)]
    if table.empty:
        return table
    clip = lambda values, n: np.clip(values, 0.5 / n, 1 - 0.5 / n)
    table["d_prime"] = norm.ppf(clip(table.hit, table.threat_n)) - norm.ppf(
        clip(table.false_alarm, table.quiet_n)
    )
    table["bias"] = -0.5 * (
        norm.ppf(clip(table.hit, table.threat_n)) + norm.ppf(clip(table.false_alarm, table.quiet_n))
    )
    return table.sort_values("d_prime", ascending=False)
