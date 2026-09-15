import math

from .regions import REGIONS, region_of
from .timeline import ParsedTimeline

SPAN = 15
BANKED = 1200.0
LOW_HEALTH = 0.45
STALE = 4
COMMIT_RANGE = 2600.0
OBJECTIVES = {"DRAGON", "HORDE"}


def _health(timeline: dict, minute: int, pid: int) -> float:
    frames = timeline.get("info", {}).get("frames") or []
    if minute >= len(frames):
        return 1.0
    payload = (frames[minute].get("participantFrames") or {}).get(str(pid)) or {}
    stats = payload.get("championStats") or {}
    top = float(stats.get("healthMax", 0.0))
    return float(stats.get("health", top)) / top if top else 1.0


def tempo_rows(
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
    deaths: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    for event in parsed.kill_events():
        minute = int(float(event.get("timestamp", 0)) // 60000)
        victim = int(event.get("victimId", 0))
        if victim in deaths and minute <= limit:
            deaths[victim].add(minute)
    contests: list[tuple[int, float, float, int]] = []
    for event in parsed.events:
        if event.get("type") != "ELITE_MONSTER_KILL":
            continue
        if str(event.get("monsterType") or "") not in OBJECTIVES:
            continue
        when = float(event.get("timestamp", 0)) / 60000.0
        position = event.get("position") or {}
        if when <= limit and position:
            killer = int(event.get("killerId", 0))
            contests.append(
                (when, float(position.get("x", 0.0)), float(position.get("y", 0.0)),
                 parsed.teams.get(killer, 0))
            )
    pressure: dict[tuple[int, int], dict] = {}
    for pid in parsed.pid_to_puuid:
        reset = 0
        banked = parsed.unspent_gold.get(pid) or []
        for minute in range(limit + 1):
            if minute in deaths[pid] or (
                minute < len(zones[pid]) and zones[pid][minute] == "BASE_OWN"
            ):
                reset = minute
            gold = banked[minute] if minute < len(banked) else 0.0
            health = _health(timeline, minute, pid)
            pressure[(pid, minute)] = {
                "time_on_map": float(minute - reset),
                "banked_gold": float(gold),
                "health": health,
                "needs_reset": float(
                    (minute - reset) >= STALE and (gold >= BANKED or health <= LOW_HEALTH)
                ),
            }
    rows = []
    for when, x, y, killer_team in contests:
        minute = int(when)
        for team in (100, 200):
            members = [pid for pid in parsed.pid_to_puuid if parsed.teams.get(pid) == team]
            for index, left in enumerate(members):
                for right in members[index + 1 :]:
                    state = {}
                    for pid, tag in ((left, "a"), (right, "b")):
                        here = parsed.position_at(pid, when)
                        near = (
                            here is not None
                            and math.hypot(here[0] - x, here[1] - y) < COMMIT_RANGE
                        )
                        info = pressure.get((pid, minute), {})
                        state[tag] = {
                            "committed": float(near),
                            "needs_reset": info.get("needs_reset", 0.0),
                            "time_on_map": info.get("time_on_map", 0.0),
                            "banked_gold": info.get("banked_gold", 0.0),
                            "health": info.get("health", 1.0),
                        }
                    misaligned = float(
                        (state["a"]["committed"] > 0 and state["b"]["needs_reset"] > 0)
                        or (state["b"]["committed"] > 0 and state["a"]["needs_reset"] > 0)
                    )
                    aligned = float(state["a"]["committed"] > 0 and state["b"]["committed"] > 0)
                    rows.append(
                        {
                            "match_id": parsed.match_id,
                            "minute": minute,
                            "team_id": team,
                            "puuid_a": parsed.pid_to_puuid[left],
                            "puuid_b": parsed.pid_to_puuid[right],
                            "pair_key": "|".join(
                                sorted((parsed.pid_to_puuid[left], parsed.pid_to_puuid[right]))
                            ),
                            "role_pair": "-".join(
                                sorted((parsed.roles.get(left, "?"), parsed.roles.get(right, "?")))
                            ),
                            "either_committed": float(
                                state["a"]["committed"] > 0 or state["b"]["committed"] > 0
                            ),
                            "aligned": aligned,
                            "misaligned": misaligned,
                            "a_needs_reset": state["a"]["needs_reset"],
                            "b_needs_reset": state["b"]["needs_reset"],
                            "max_time_on_map": max(
                                state["a"]["time_on_map"], state["b"]["time_on_map"]
                            ),
                            "max_banked": max(state["a"]["banked_gold"], state["b"]["banked_gold"]),
                            "min_health": min(state["a"]["health"], state["b"]["health"]),
                            "we_took_it": int(team == killer_team),
                        }
                    )
    return rows
