import math

from .timeline import ParsedTimeline

SPAN = 15
NEAR = 2500.0
REACH = 5000.0
LATENCY_WINDOW = 3
MONSTERS = {"DRAGON", "HORDE"}  # herald spawns at 15:00, baron later; both outside the window


def _position(event: dict) -> tuple[float, float] | None:
    payload = event.get("position") or {}
    if "x" not in payload:
        return None
    return float(payload["x"]), float(payload["y"])


def _trigger(
    kind: str, event: dict, parsed: ParsedTimeline, detail: str = "", index: int | None = None
) -> dict | None:
    spot = _position(event)
    if spot is None:
        return None
    actor = next((int(event[k]) for k in ("killerId", "creatorId", "participantId") if event.get(k)), 0)
    return {
        "kind": kind,
        "detail": detail,
        "event_index": index,
        "minute": float(event.get("timestamp", 0)) / 60000.0,
        "actor": actor,
        "actor_team": parsed.teams.get(actor, int(event.get("killerTeamId", 0) or 0)),
        "victim": int(event.get("victimId", 0)),
        "x": spot[0],
        "y": spot[1],
    }


def triggers(parsed: ParsedTimeline, limit: int) -> list[dict]:
    out = []
    for index, event in enumerate(parsed.events):
        minute = float(event.get("timestamp", 0)) / 60000.0
        if minute > limit:
            continue
        kind = event.get("type")
        if kind == "ELITE_MONSTER_KILL" and str(event.get("monsterType")) in MONSTERS:
            row = _trigger("objective", event, parsed, str(event.get("monsterType")), index)
        elif kind == "CHAMPION_KILL":
            row = _trigger("kill", event, parsed, "", index)
        elif kind == "TURRET_PLATE_DESTROYED":
            row = _trigger("plate", event, parsed, "", index)
        elif kind == "BUILDING_KILL":
            row = _trigger("building", event, parsed, str(event.get("towerType") or ""), index)
        else:
            continue
        if row is not None:
            out.append(row)
    for index, row in enumerate(sorted(out, key=lambda item: item["minute"])):
        row["trigger_id"] = index
    return out


def _reaction(parsed: ParsedTimeline, pid: int, trigger: dict, limit: int) -> dict:
    track = parsed.positions.get(pid) or []
    when = float(trigger["minute"])
    minute = int(when)
    spot = (trigger["x"], trigger["y"])
    if not parsed.alive_at(pid, when):
        return {}
    source = trigger.get("event_index")
    here = parsed.position_at(pid, when, ignore=source)
    earlier = parsed.position_at(pid, max(when - 1.0, 0.0), ignore=source)
    if here is None or earlier is None or minute >= len(track):
        return {}
    approach = math.hypot(earlier[0] - spot[0], earlier[1] - spot[1])
    arrival = math.hypot(here[0] - spot[0], here[1] - spot[1])
    closing, latency, departure = arrival, float(LATENCY_WINDOW), arrival
    for step in range(1, LATENCY_WINDOW + 1):
        index = minute + step
        if index >= len(track) or index > limit:
            break
        distance = math.hypot(track[index][0] - spot[0], track[index][1] - spot[1])
        closing = min(closing, distance)
        departure = distance
        if distance < NEAR and latency == float(LATENCY_WINDOW):
            latency = float(step)
    return {
        "approach": approach / 1000.0,
        "arrival": arrival / 1000.0,
        "closest_after": closing / 1000.0,
        "present": float(arrival < NEAR),
        "nearby": float(arrival < REACH),
        "converged": float(approach >= REACH and closing < NEAR),
        "left_after": float(departure - arrival > NEAR),
        "held_ground": float(arrival < NEAR and departure < NEAR),
        "latency": latency,
    }


def event_response_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    rows = []
    for trigger in triggers(parsed, limit):
        for pid, puuid in parsed.pid_to_puuid.items():
            reaction = _reaction(parsed, pid, trigger, limit)
            if not reaction:
                continue
            team = parsed.teams.get(pid, 100)
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "trigger_id": trigger["trigger_id"],
                    "trigger": trigger["kind"],
                    "detail": trigger["detail"],
                    "minute": trigger["minute"],
                    "puuid": puuid,
                    "role": parsed.roles.get(pid, "") or "UNKNOWN",
                    "team_id": team,
                    "ours": int(team == trigger["actor_team"]) if trigger["actor_team"] else 0,
                    "is_actor": int(pid == trigger["actor"]),
                    "is_victim": int(pid == trigger["victim"]),
                    **reaction,
                }
            )
    return rows


def dyadic_responses(responses, same_team: bool = True):
    import pandas as pd

    frame = responses if isinstance(responses, pd.DataFrame) else pd.DataFrame(responses)
    if frame.empty:
        return frame
    keys = ["match_id", "trigger_id", "trigger", "detail", "minute"]
    joined = frame.merge(frame, on=keys, suffixes=("_a", "_b"))
    joined = joined[joined["puuid_a"] < joined["puuid_b"]]
    if same_team:
        joined = joined[joined["team_id_a"] == joined["team_id_b"]]
    joined["pair_key"] = joined["puuid_a"] + "|" + joined["puuid_b"]
    joined["role_pair"] = [
        "-".join(sorted(pair)) for pair in zip(joined["role_a"], joined["role_b"])
    ]
    joined["both_present"] = ((joined["present_a"] > 0) & (joined["present_b"] > 0)).astype(float)
    joined["either_present"] = ((joined["present_a"] > 0) | (joined["present_b"] > 0)).astype(float)
    joined["both_converged"] = (
        (joined["converged_a"] > 0) & (joined["converged_b"] > 0)
    ).astype(float)
    joined["latency_gap"] = (joined["latency_a"] - joined["latency_b"]).abs()
    return joined.reset_index(drop=True)


RESPONSE_COLUMNS = [
    "approach",
    "arrival",
    "closest_after",
    "present",
    "nearby",
    "converged",
    "left_after",
    "held_ground",
    "latency",
]
TRIGGER_KINDS = ("objective", "kill", "plate", "building")
