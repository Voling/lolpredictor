import math

from .regions import MAP_SPAN, REGIONS, region_of
from .timeline import VISION_WARDS, ParsedTimeline
from .wave import LANE_PREFIX, wave_states

FRAME_MS = 60000.0
GANK_RADIUS = 2200.0
INVADE_LEAD = 90000.0
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
RIVER = {"RIVER_BARON", "RIVER_DRAGON"}


def interpolate(track: list[tuple[float, float]], timestamp: float) -> tuple[float, float] | None:
    if not track:
        return None
    exact = timestamp / FRAME_MS
    low = int(math.floor(exact))
    if low >= len(track) - 1:
        return track[-1]
    if low < 0:
        return track[0]
    weight = exact - low
    before, after = track[low], track[low + 1]
    return (
        before[0] + (after[0] - before[0]) * weight,
        before[1] + (after[1] - before[1]) * weight,
    )


def _jungler_events(parsed: ParsedTimeline, jungler: int | None, span: int) -> tuple[list[float], list[dict]]:
    invades: list[float] = []
    ganks: list[dict] = []
    if jungler is None:
        return invades, ganks
    team = parsed.teams.get(jungler, 100)
    track = parsed.positions.get(jungler) or []
    previous = None
    for minute in range(min(len(track), span + 1)):
        zone = REGIONS[region_of(track[minute][0], track[minute][1], team)]
        if zone in ENEMY_JUNGLE and previous not in ENEMY_JUNGLE:
            invades.append(minute * FRAME_MS)
        previous = zone
    for event in parsed.kill_events():
        if jungler not in parsed.involved(event):
            continue
        position = event.get("position") or {}
        if not position:
            continue
        ganks.append(
            {
                "timestamp": float(event.get("timestamp", 0)),
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
            }
        )
    return invades, ganks


def ward_rows(
    match: dict, timeline: dict, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = len(parsed.minutes) - 1
    lane_states = {
        pid: wave_states(parsed, pid, span)
        for pid in parsed.pid_to_puuid
        if parsed.roles.get(pid) in LANE_PREFIX
    }
    junglers = {team: parsed.role_pid(team, "JUNGLE") for team in (100, 200)}
    jungler_moves = {team: _jungler_events(parsed, junglers[team], span) for team in (100, 200)}

    rows = []
    for event in parsed.events:
        if event.get("type") != "WARD_PLACED" or event.get("wardType") not in VISION_WARDS:
            continue
        pid = int(event.get("creatorId", 0))
        puuid = parsed.pid_to_puuid.get(pid)
        if puuid is None:
            continue
        timestamp = float(event.get("timestamp", 0))
        if timestamp / FRAME_MS > span:
            continue
        team = parsed.teams.get(pid, 100)
        spot = interpolate(parsed.positions.get(pid) or [], timestamp)
        if spot is None:
            continue
        zone = REGIONS[region_of(spot[0], spot[1], team)]
        invades, ganks = jungler_moves[team]
        next_invade = min(
            (moment - timestamp for moment in invades if moment >= timestamp), default=None
        )
        nearest_gank = min(
            (
                (math.hypot(spot[0] - gank["x"], spot[1] - gank["y"]), gank["timestamp"] - timestamp)
                for gank in ganks
                if gank["timestamp"] >= timestamp
            ),
            default=(None, None),
        )
        own_lane = lane_states.get(pid)
        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "role": parsed.roles.get(pid, "") or "UNKNOWN",
                "team_id": team,
                "timestamp_ms": int(timestamp),
                "clock": f"{int(timestamp // 60000)}:{int(timestamp % 60000 // 1000):02d}",
                "minute": timestamp / FRAME_MS,
                "ward_type": event.get("wardType"),
                "x": round(spot[0], 1),
                "y": round(spot[1], 1),
                "zone": zone,
                "in_enemy_jungle": zone in ENEMY_JUNGLE,
                "in_own_jungle": zone in OWN_JUNGLE,
                "in_river": zone in RIVER,
                "own_lane_state": own_lane[int(timestamp // 60000)]
                if own_lane and int(timestamp // 60000) < len(own_lane)
                else None,
                "seconds_before_jungler_invade": None
                if next_invade is None
                else round(next_invade / 1000.0, 1),
                "warded_before_invade": bool(next_invade is not None and next_invade <= INVADE_LEAD),
                "distance_to_next_gank": None
                if nearest_gank[0] is None
                else round(nearest_gank[0] / MAP_SPAN, 3),
                "warded_near_next_gank": bool(
                    nearest_gank[0] is not None and nearest_gank[0] <= GANK_RADIUS
                ),
            }
        )
    return rows
