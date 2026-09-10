import math

from .regions import REGIONS, region_of
from .timeline import ParsedTimeline
from .wave import wave_states, PUSH, DEFENSIVE, NEUTRAL

SPAN = 15
MEMORY = 2
SIGHT = 1350.0
GROUP_RANGE = 2200.0
LANE_PREFIX = {"TOP": "LANE_TOP", "MIDDLE": "LANE_MID", "BOTTOM": "LANE_BOT", "UTILITY": "LANE_BOT"}
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
RIVER = {"RIVER_BARON", "RIVER_DRAGON"}
ACTIONS = ("recall", "shove", "freeze", "lane_neutral", "roam", "invade", "jungle", "river")
STATES = ("died_solo", "died_with_allies", "jungler_nearby", "enemy_nearby",
          "behind", "ahead", "even")


WAVE_ACTION = {PUSH: "shove", DEFENSIVE: "freeze", NEUTRAL: "lane_neutral"}


def _action(zone: str, previous: str, role: str, wave: str) -> str:
    if zone == "BASE_OWN":
        return "recall"
    if zone in ENEMY_JUNGLE:
        return "invade"
    if zone in RIVER:
        return "river"
    if zone in OWN_JUNGLE:
        return "jungle"
    home = LANE_PREFIX.get(role)
    if home and zone.startswith(home):
        return WAVE_ACTION.get(wave, "lane_neutral")
    if zone.startswith("LANE_"):
        return "roam"
    return WAVE_ACTION.get(wave, "lane_neutral") if zone == previous else "roam"


def policy_rows(match: dict, timeline: dict, span: int = SPAN) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
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
    deaths: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    for event in parsed.kill_events():
        minute = int(float(event.get("timestamp", 0)) // 60000)
        victim = int(event.get("victimId", 0))
        if victim in deaths and minute <= limit:
            deaths[victim].add(minute)
    gold = {pid: (parsed.gold.get(pid) or []) for pid in parsed.pid_to_puuid}
    counterpart = {}
    for pid in parsed.pid_to_puuid:
        role, team = parsed.roles.get(pid), parsed.teams.get(pid)
        counterpart[pid] = next(
            (
                other
                for other in parsed.pid_to_puuid
                if parsed.roles.get(other) == role and parsed.teams.get(other) != team
            ),
            None,
        )
    waves = {pid: wave_states(parsed, pid, limit) for pid in parsed.pid_to_puuid}
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        team = parsed.teams.get(pid, 100)
        role = parsed.roles.get(pid, "") or "UNKNOWN"
        track = parsed.positions.get(pid) or []
        mine = zones.get(pid, [])
        allies = [other for other in parsed.pid_to_puuid if parsed.teams.get(other) == team and other != pid]
        enemies = [other for other in parsed.pid_to_puuid if parsed.teams.get(other) != team]
        rival = counterpart.get(pid)
        last_reset = 0
        for minute in range(1, min(limit, len(mine) - 1)):
            if minute in deaths[pid] or mine[minute] == "BASE_OWN":
                last_reset = minute
            if minute in deaths[pid]:
                continue
            recent = range(max(minute - MEMORY, 0), minute + 1)
            died = any(step in deaths[pid] for step in recent)
            ally_died = any(step in deaths[other] for other in allies for step in recent)
            seen, jungler_seen = False, False
            if minute < len(track):
                for other in enemies:
                    other_track = parsed.positions.get(other) or []
                    if minute < len(other_track) and math.dist(track[minute], other_track[minute]) < SIGHT:
                        seen = True
                        if parsed.roles.get(other) == "JUNGLE":
                            jungler_seen = True
            lead = 0.0
            if rival is not None and minute < len(gold[pid]) and minute < len(gold.get(rival, [])):
                lead = gold[pid][minute] - gold[rival][minute]
            solo_death = False
            if died:
                when = next(step for step in recent if step in deaths[pid])
                near = sum(
                    1
                    for other in allies
                    if when < len(parsed.positions.get(other) or [])
                    and when < len(track)
                    and math.dist(track[when], parsed.positions[other][when]) < GROUP_RANGE
                )
                solo_death = near == 0
            if died and solo_death:
                state = "died_solo"
            elif died:
                state = "died_with_allies"
            elif jungler_seen:
                state = "jungler_nearby"
            elif seen:
                state = "enemy_nearby"
            elif lead > 400:
                state = "ahead"
            elif lead < -400:
                state = "behind"
            else:
                state = "even"
            grouped = sum(
                1
                for other in allies
                if minute < len(parsed.positions.get(other) or [])
                and minute < len(track)
                and math.dist(track[minute], parsed.positions[other][minute]) < GROUP_RANGE
            )
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "puuid": puuid,
                    "role": role,
                    "team_id": team,
                    "minute": minute,
                    "state": state,
                    "died_recently": float(died),
                    "ally_died_recently": float(ally_died),
                    "enemy_near": float(seen),
                    "jungler_nearby": float(jungler_seen),
                    "lane_lead": lead / 1000.0,
                    "allies_nearby": float(grouped),
                    "time_on_map": float(minute - last_reset),
                    "action": _action(
                        mine[minute + 1], mine[minute], role,
                        waves[pid][minute + 1] if minute + 1 < len(waves[pid]) else NEUTRAL,
                    ),
                    "previous": _action(
                        mine[minute], mine[max(minute - 1, 0)], role,
                        waves[pid][minute] if minute < len(waves[pid]) else NEUTRAL,
                    ),
                }
            )
    return rows


POLICY_COLUMNS = [
    "died_recently",
    "ally_died_recently",
    "enemy_near",
    "jungler_nearby",
    "lane_lead",
    "allies_nearby",
    "time_on_map",
]
