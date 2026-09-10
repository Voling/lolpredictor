import math

from .timeline import ParsedTimeline

SPAN = 15
NEARBY = 2500.0
APPROACH = 5000.0
CONTEST_WINDOW = 1.0
AFTER_WINDOW = 1.5
PITS = {
    "DRAGON": {100: (9866, 4414), 200: (9866, 4414)},
    "HORDE": {100: (4300, 10200), 200: (4300, 10200)},
}
OBJECTIVE_KINDS = ("DRAGON", "HORDE")
STATES = ("contested", "taken_by_us", "taken_by_them")


def _distance(point: tuple[float, float], pit: tuple[float, float]) -> float:
    return math.hypot(point[0] - pit[0], point[1] - pit[1])


def objective_events(parsed: ParsedTimeline, limit: int) -> list[dict]:
    out = []
    for event in parsed.events:
        if event.get("type") != "ELITE_MONSTER_KILL":
            continue
        minute = float(event.get("timestamp", 0)) / 60000.0
        kind = str(event.get("monsterType") or "")
        if minute > limit or kind not in PITS:
            continue
        position = event.get("position") or {}
        killer = int(event.get("killerId", 0))
        out.append(
            {
                "minute": minute,
                "kind": kind,
                "subtype": event.get("monsterSubType"),
                "killer_team": parsed.teams.get(killer, int(event.get("killerTeamId", 0) or 0)),
                "x": float(position.get("x", PITS[kind][100][0])),
                "y": float(position.get("y", PITS[kind][100][1])),
            }
        )
    return sorted(out, key=lambda row: row["minute"])


def objective_rows(match: dict, timeline: dict, span: int = SPAN) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    events = objective_events(parsed, limit)
    if not events:
        return []
    kills = [
        event
        for event in parsed.kill_events()
        if float(event.get("timestamp", 0)) / 60000.0 <= limit
    ]
    rows = []
    for event in events:
        minute = int(event["minute"])
        before = max(minute - 1, 0)
        pit = (event["x"], event["y"])
        for pid, puuid in parsed.pid_to_puuid.items():
            track = parsed.positions.get(pid) or []
            if before >= len(track) or minute >= len(track):
                continue
            team = parsed.teams.get(pid, 100)
            approach = _distance(track[before], pit)
            arrival = _distance(track[minute], pit)
            after = (
                _distance(track[min(minute + 1, len(track) - 1)], pit)
                if minute + 1 < len(track)
                else arrival
            )
            involved = any(
                pid in parsed.involved(k)
                and abs(float(k.get("timestamp", 0)) / 60000.0 - event["minute"]) <= CONTEST_WINDOW
                and _distance(
                    (
                        float((k.get("position") or {}).get("x", 0.0)),
                        float((k.get("position") or {}).get("y", 0.0)),
                    ),
                    pit,
                )
                < APPROACH
                for k in kills
            )
            died = any(
                int(k.get("victimId", 0)) == pid
                and abs(float(k.get("timestamp", 0)) / 60000.0 - event["minute"]) <= CONTEST_WINDOW
                for k in kills
            )
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "puuid": puuid,
                    "team_id": team,
                    "role": parsed.roles.get(pid, "") or "UNKNOWN",
                    "objective": event["kind"],
                    "subtype": event["subtype"],
                    "minute": event["minute"],
                    "ours": int(team == event["killer_team"]),
                    "o_approach_distance": approach / 1000.0,
                    "o_arrival_distance": arrival / 1000.0,
                    "o_present": float(arrival < NEARBY),
                    "o_approaching": float(approach < APPROACH),
                    "o_committed": float(arrival < NEARBY and approach < APPROACH),
                    "o_rotated_in": float(approach >= APPROACH and arrival < NEARBY),
                    "o_left_after": float(after - arrival > NEARBY),
                    "o_fought": float(involved),
                    "o_died": float(died),
                }
            )
    return rows


OBJECTIVE_FEATURE_COLUMNS = [
    "o_approach_distance",
    "o_arrival_distance",
    "o_present",
    "o_approaching",
    "o_committed",
    "o_rotated_in",
    "o_left_after",
    "o_fought",
    "o_died",
]
