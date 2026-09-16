TRIGGERS = {"kill": "kill", "plate": "plate", "objective": "objective", "building": "building"}
SIDES = {"ours": "ally", "theirs": "enemy"}
DISTANCE = {"near": "nearby", "mid": "at mid range", "far": "from far away"}
RESPONSES = {
    "converged": "converged on",
    "held": "held ground at",
    "left": "left after",
    "present": "present at",
    "absent": "absent from",
}
OBJECTIVES = {"DRAGON": "dragon", "HORDE": "void grubs"}
OBJECTIVE_RESPONSES = {
    "died": "died at",
    "fought": "fought at",
    "committed": "committed to",
    "rotated": "rotated toward",
    "approached": "approached",
    "absent": "absent from",
}
WARD_TIMES = {"early": "before 5 minutes", "mid": "between 5 and 10 minutes", "late": "after 10 minutes"}
WARD_ZONES = {
    "lane_own_side": "on own side of the lane",
    "lane_middle": "in the middle of the lane",
    "lane_enemy_side": "on the enemy side of the lane",
    "own_jungle": "in own jungle",
    "enemy_jungle": "in the enemy jungle",
    "river": "in the river",
    "other": "elsewhere",
}
OPENINGS = {"gank": "first gank", "invade": "first invade"}
OPENING_BANDS = {"by3": "by 3 minutes", "by5": "by 5 minutes", "by8": "by 8 minutes", "late": "after 8 minutes", "never": "not before 15 minutes"}
JUNGLE_SIDES = {"crossed": "crossed to the other side of the jungle", "stayed": "stayed on one side of the jungle"}
LANE_BANDS = {
    "deep_own": "deep in own half",
    "own": "on own side",
    "mid": "at the middle",
    "theirs": "on the enemy side",
    "deep_theirs": "deep in the enemy half",
}
PRIORITY_TIMES = {"all": "", "pre_objective": " in the minute before an objective"}
KINDS = {
    "initiate": "starts a fight",
    "follow": "follows into a fight",
    "defend": "defends",
    "dive": "dives",
    "fight_join": "joins a fight",
    "overstay": "overstays with a full bank",
    "greed_punished": "punished for greed",
    "ward_pre_objective": "wards before an objective",
    "ward_clear": "clears wards",
}
KIND_SITUATIONS = {
    "initiate": "starting fights",
    "follow": "following into fights",
    "defend": "defending",
    "dive": "diving",
    "fight_join": "joining fights",
    "overstay": "overstaying with a full bank",
    "greed_punished": "getting punished for greed",
    "ward_pre_objective": "warding before objectives",
    "ward_clear": "clearing wards",
}
LANES = {"t": "top", "m": "mid", "b": "bot"}
HALVES = {"own": "on own half", "away": "on the enemy half"}
COMPONENTS = {"habit": "habit component", "move": "movement component", "style": "embedding component"}


def _priority(lanes: str) -> str:
    held = [LANES[letter] for letter in lanes if letter in LANES]
    if not held:
        return "with no lane holding priority"
    if len(held) == 1:
        return f"with {held[0]} holding priority"
    return f"with {', '.join(held[:-1])} and {held[-1]} holding priority"


def describe(cell: str) -> str:
    parts = cell.split("_")
    prefix = parts[0]
    if prefix == "rsp" and len(parts) == 5:
        _, trigger, side, band, outcome = parts
        return f"{RESPONSES[outcome]} an {SIDES[side]} {TRIGGERS[trigger]} {DISTANCE[band]}"
    if prefix == "obj" and len(parts) == 5:
        _, objective, side, band, outcome = parts
        return f"{OBJECTIVE_RESPONSES[outcome]} an {SIDES[side]} {OBJECTIVES[objective]} {DISTANCE[band]}"
    if prefix == "ward" and len(parts) >= 3:
        zone = "_".join(parts[2:])
        return f"ward placed {WARD_TIMES[parts[1]]} {WARD_ZONES[zone]}"
    if prefix == "jgl" and parts[1] == "sides":
        return JUNGLE_SIDES[parts[2]]
    if prefix == "jgl":
        return f"{OPENINGS[parts[1]]} {OPENING_BANDS[parts[2]]}"
    if prefix == "prio":
        situation = "pre_objective" if parts[1] == "pre" else "all"
        outcome = "_".join(parts[3:] if situation == "pre_objective" else parts[2:])
        when = PRIORITY_TIMES[situation]
        if outcome == "off_lane":
            return f"away from lane{when}"
        if outcome == "dead":
            return f"dead{when}"
        own, opponent, farming = _lane_outcome(outcome)
        where = f"standing {LANE_BANDS[own]}"
        against = "opponent away" if opponent == "absent" else f"opponent {LANE_BANDS[opponent]}"
        return f"{where}, {against}, {'farming' if farming == 'farm' else 'not farming'}{when}"
    if prefix == "tend" and len(parts) >= 4:
        kind = "_".join(parts[1:-2])
        return f"{KINDS[kind]} {_priority(parts[-2])}, {HALVES[parts[-1]]}"
    if prefix in COMPONENTS:
        index = parts[-1].removeprefix("e")
        return f"{COMPONENTS[prefix]} {index}"
    return cell


def _lane_outcome(outcome: str) -> tuple[str, str, str]:
    for own in sorted(LANE_BANDS, key=len, reverse=True):
        if outcome.startswith(own + "_"):
            rest = outcome[len(own) + 1 :]
            for opponent in sorted([*LANE_BANDS, "absent"], key=len, reverse=True):
                if rest.startswith(opponent + "_"):
                    return own, opponent, rest[len(opponent) + 1 :]
    raise ValueError(f"unknown lane outcome {outcome!r}")


def situation_of(cell: str) -> str:
    parts = cell.split("_")
    prefix = parts[0]
    if prefix in ("rsp", "obj") and len(parts) == 5:
        return "_".join(parts[:4])
    if prefix == "ward":
        return "_".join(parts[:2])
    if prefix == "jgl":
        return "_".join(parts[:2])
    if prefix == "prio":
        return "prio_pre_objective" if parts[1] == "pre" else "prio_all"
    if prefix == "tend":
        return "_".join(parts[:-2])
    return prefix


def describe_situation(situation: str) -> str:
    parts = situation.split("_")
    prefix = parts[0]
    if prefix == "rsp" and len(parts) == 4:
        return f"after an {SIDES[parts[2]]} {TRIGGERS[parts[1]]} {DISTANCE[parts[3]]}"
    if prefix == "obj" and len(parts) == 4:
        return f"at an {SIDES[parts[2]]} {OBJECTIVES[parts[1]]} {DISTANCE[parts[3]]}"
    if prefix == "ward":
        return f"wards {WARD_TIMES[parts[1]]}"
    if prefix == "jgl":
        return {"gank": "first gank timing", "invade": "first invade timing", "sides": "jungle side choice"}[parts[1]]
    if prefix == "prio":
        return "lane state" + (PRIORITY_TIMES["pre_objective"] if situation.endswith("pre_objective") else "")
    if prefix == "tend":
        return KIND_SITUATIONS["_".join(parts[1:])]
    return {"habit": "habit components", "move": "movement components", "style": "embedding components"}.get(prefix, situation)


def named(cell: str) -> bool:
    return cell.split("_")[0] not in COMPONENTS
