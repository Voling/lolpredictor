from .regions import MAP_SPAN, REGIONS, region_of
from .timeline import ParsedTimeline

LANE_PREFIX = {"TOP": "LANE_TOP", "MIDDLE": "LANE_MID", "BOTTOM": "LANE_BOT", "UTILITY": "LANE_BOT"}
PUSH_PROGRESS = 1.0
DEFENSIVE_PROGRESS = 0.85
WAVE_CS = 4.0
SUSTAINED_MINUTES = 3
RESPONSE_WINDOW = 2
EARLY_PUSH_MINUTE = 4
MINIONS_PER_MINUTE = 13.0
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
ACTIONS = ("invade", "join_lane", "objective", "recall", "hold")

PUSH = "push"
DEFENSIVE = "defensive"
NEUTRAL = "neutral"
OFF_LANE = "off_lane"


def lane_progress(x: float, y: float, team: int) -> float:
    progress = (x + y) / MAP_SPAN
    return 2.0 - progress if team == 200 else progress


def wave_states(parsed: ParsedTimeline, pid: int, span: int) -> list[str]:
    role = parsed.roles.get(pid, "")
    team = parsed.teams.get(pid, 100)
    prefix = LANE_PREFIX.get(role)
    positions = parsed.positions.get(pid) or []
    cs = parsed.cs.get(pid) or []
    states = []
    for minute in range(min(len(positions), span + 1)):
        x, y = positions[minute]
        zone = REGIONS[region_of(x, y, team)]
        if prefix is None or not zone.startswith(prefix):
            states.append(OFF_LANE)
            continue
        farmed = (cs[minute] - cs[minute - 1]) if 0 < minute < len(cs) else 0.0
        progress = lane_progress(x, y, team)
        if progress >= PUSH_PROGRESS and farmed >= WAVE_CS:
            states.append(PUSH)
        elif progress <= DEFENSIVE_PROGRESS and farmed >= WAVE_CS:
            states.append(DEFENSIVE)
        else:
            states.append(NEUTRAL)
    return states


def _sustained(states: list[str], target: str, run: int) -> int:
    best = streak = 0
    for state in states:
        streak = streak + 1 if state == target else 0
        best = max(best, streak)
    return int(best >= run)


def wave_rows(match: dict, timeline: dict) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = len(parsed.minutes) - 1
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        role = parsed.roles.get(pid, "")
        if role not in LANE_PREFIX:
            continue
        states = wave_states(parsed, pid, span)
        cs = parsed.cs.get(pid) or []
        lane_minutes = [index for index, state in enumerate(states) if state != OFF_LANE]
        pushes = [index for index, state in enumerate(states) if state == PUSH]
        defensive = [index for index, state in enumerate(states) if state == DEFENSIVE]
        first_push = pushes[0] if pushes else span
        jungle = parsed.jungle_cs.get(pid) or []
        captured = [
            (cs[index] - cs[index - 1])
            - ((jungle[index] - jungle[index - 1]) if 0 < index < len(jungle) else 0.0)
            for index in pushes
            if 0 < index < len(cs)
        ]
        total = max(len(lane_minutes), 1)
        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "role": role,
                "w_first_push_minute": float(first_push),
                "w_pushed_by_4": float(bool(pushes) and first_push <= EARLY_PUSH_MINUTE),
                "w_pushed_at_all": float(bool(pushes)),
                "w_push_share": len(pushes) / total,
                "w_defensive_share": len(defensive) / total,
                "w_push_cs_per_minute": sum(captured) / len(captured) if captured else 0.0,
                "w_push_minion_share": (sum(captured) / len(captured) / MINIONS_PER_MINUTE)
                if captured
                else 0.0,
                "w_sustained_defensive": float(_sustained(states, DEFENSIVE, SUSTAINED_MINUTES)),
                "w_lane_minutes": float(len(lane_minutes)),
            }
        )
    return rows


def _action(parsed: ParsedTimeline, pid: int, team: int, lane_role: str, start: int, span: int) -> str:
    positions = parsed.positions.get(pid) or []
    prefix = LANE_PREFIX.get(lane_role, "")
    objectives = {
        int(float(event.get("timestamp", 0)) // 60000)
        for event in parsed.events
        if event.get("type") == "ELITE_MONSTER_KILL" and pid in parsed.involved(event)
    }
    for minute in range(start, min(start + RESPONSE_WINDOW + 1, span + 1)):
        if minute >= len(positions):
            break
        zone = REGIONS[region_of(positions[minute][0], positions[minute][1], team)]
        if zone in ENEMY_JUNGLE:
            return "invade"
        if minute in objectives:
            return "objective"
        if zone.startswith(prefix) and lane_progress(*positions[minute], team) >= PUSH_PROGRESS:
            return "join_lane"
        if zone == "BASE_OWN":
            return "recall"
    return "hold"


def response_rows(match: dict, timeline: dict) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = len(parsed.minutes) - 1
    states = {pid: wave_states(parsed, pid, span) for pid in parsed.pid_to_puuid}
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        role = parsed.roles.get(pid, "")
        if role not in LANE_PREFIX:
            continue
        team = parsed.teams.get(pid, 100)
        pushed = [
            minute
            for minute, state in enumerate(states[pid])
            if state == PUSH and (minute == 0 or states[pid][minute - 1] != PUSH)
        ]
        for minute in pushed:
            for other, other_puuid in parsed.pid_to_puuid.items():
                if other == pid or parsed.teams.get(other) != team:
                    continue
                rows.append(
                    {
                        "match_id": parsed.match_id,
                        "pusher": puuid,
                        "responder": other_puuid,
                        "pusher_role": role,
                        "responder_role": parsed.roles.get(other, "") or "UNKNOWN",
                        "minute": float(minute),
                        "action": _action(parsed, other, team, role, minute, span),
                    }
                )
    return rows


WAVE_FEATURE_COLUMNS = [
    "w_first_push_minute",
    "w_pushed_by_4",
    "w_pushed_at_all",
    "w_push_share",
    "w_defensive_share",
    "w_push_cs_per_minute",
    "w_push_minion_share",
    "w_sustained_defensive",
]
