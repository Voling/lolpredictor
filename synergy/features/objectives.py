import math

from .timeline import ParsedTimeline

from ..config import get_settings

SPAN = get_settings().feature_minutes
NEARBY = 2500.0
APPROACH = 5000.0
CONTEST_WINDOW = 1.0
ATTEMPT_GAP = 0.75
PITS = {"DRAGON": (9846.0, 4421.0), "HORDE": (4790.0, 10180.0), "RIFTHERALD": (4720.0, 10000.0)}
SPAWNS = {"DRAGON": 5.0, "HORDE": 8.0, "RIFTHERALD": 15.0}
RESPAWN = {"DRAGON": 5.0}
GONE = {"HORDE": 15.0}
CAMPS = {"HORDE": 3}
SIDES = ("ours", "theirs", "none")


def _distance(point: tuple[float, float], pit: tuple[float, float]) -> float:
    return math.hypot(point[0] - pit[0], point[1] - pit[1])


def _minute(event: dict) -> float:
    return float(event.get("timestamp", 0)) / 60000.0


def _point(event: dict, fallback: tuple[float, float]) -> tuple[float, float]:
    position = event.get("position") or {}
    return float(position.get("x", fallback[0])), float(position.get("y", fallback[1]))


def up_windows(kind: str, takes: list[float], limit: float) -> list[tuple[float, float]]:
    windows, start, count = [], SPAWNS[kind], 0
    for take in sorted(takes):
        count += 1
        if count % CAMPS.get(kind, 1):
            continue
        windows.append((start, take))
        if kind not in RESPAWN:
            return [(low, high) for low, high in windows if high > low]
        start = take + RESPAWN[kind]
    windows.append((start, min(limit, GONE.get(kind, limit))))
    return [(low, high) for low, high in windows if high > low]


def _majority(teams: list[int]) -> int | None:
    if not teams:
        return None
    counts = {team: teams.count(team) for team in teams}
    best = max(counts.values())
    return next(team for team in reversed(teams) if counts[team] == best)


def attempts(kind: str, takes: list[dict], kills: list[dict], limit: float) -> list[dict]:
    pit = PITS[kind]
    windows = up_windows(kind, [take["minute"] for take in takes], limit)
    activity = [(take["minute"], take) for take in takes]
    for kill in kills:
        when = _minute(kill)
        if any(low <= when <= high for low, high in windows) and _distance(_point(kill, (0.0, 0.0)), pit) < NEARBY:
            activity.append((when, None))
    activity.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, dict | None]]] = []
    for when, take in activity:
        if groups and when - groups[-1][-1][0] <= ATTEMPT_GAP:
            groups[-1].append((when, take))
        else:
            groups.append([(when, take)])
    found = []
    for group in groups:
        taken = [take for _, take in group if take is not None]
        last = taken[-1] if taken else None
        found.append(
            {
                "kind": kind,
                "subtype": last["subtype"] if last else None,
                "start": group[0][0],
                "end": group[-1][0],
                "minute": last["minute"] if last else group[-1][0],
                "killer_team": _majority([take["killer_team"] for take in taken]),
                "x": last["x"] if last else pit[0],
                "y": last["y"] if last else pit[1],
            }
        )
    return found


def objective_events(parsed: ParsedTimeline, limit: float) -> list[dict]:
    takes: dict[str, list[dict]] = {kind: [] for kind in PITS}
    for event in parsed.events:
        if event.get("type") != "ELITE_MONSTER_KILL":
            continue
        minute = _minute(event)
        kind = str(event.get("monsterType") or "")
        if minute > limit or kind not in PITS:
            continue
        killer = int(event.get("killerId", 0))
        x, y = _point(event, PITS[kind])
        takes[kind].append(
            {
                "minute": minute,
                "subtype": event.get("monsterSubType"),
                "killer_team": parsed.teams.get(killer, int(event.get("killerTeamId", 0) or 0)),
                "x": x,
                "y": y,
            }
        )
    kills = [event for event in parsed.kill_events() if _minute(event) <= limit]
    found = [attempt for kind in PITS for attempt in attempts(kind, takes[kind], kills, limit)]
    return sorted(found, key=lambda row: row["minute"])


def objective_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    events = objective_events(parsed, limit)
    if not events:
        return []
    kills = [event for event in parsed.kill_events() if _minute(event) <= limit]
    rows = []
    for event in events:
        when, start = float(event["minute"]), float(event["start"])
        minute = int(when)
        pit = (event["x"], event["y"])
        nearby = [kill for kill in kills if start - CONTEST_WINDOW <= _minute(kill) <= float(event["end"]) + CONTEST_WINDOW]
        for pid, puuid in parsed.pid_to_puuid.items():
            track = parsed.positions.get(pid) or []
            here = parsed.position_at(pid, when)
            earlier = parsed.position_at(pid, max(start - 1.0, 0.0))
            if here is None or earlier is None or minute >= len(track):
                continue
            team = parsed.teams.get(pid, 100)
            approach = _distance(earlier, pit)
            arrival = _distance(here, pit)
            after = (
                _distance(track[min(minute + 1, len(track) - 1)], pit)
                if minute + 1 < len(track)
                else arrival
            )
            involved = any(pid in parsed.involved(kill) and _distance(_point(kill, (0.0, 0.0)), pit) < APPROACH for kill in nearby)
            died = any(int(kill.get("victimId", 0)) == pid for kill in nearby)
            side = "none" if event["killer_team"] is None else ("ours" if team == event["killer_team"] else "theirs")
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "puuid": puuid,
                    "team_id": team,
                    "role": parsed.roles.get(pid, "") or "UNKNOWN",
                    "objective": event["kind"],
                    "subtype": event["subtype"],
                    "minute": event["minute"],
                    "ours": int(side == "ours"),
                    "side": side,
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
