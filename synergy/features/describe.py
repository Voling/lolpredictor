TRIGGERS = {
    "kill": ("gets a kill", "got a kill"),
    "plate": ("takes a plate", "took a plate"),
    "objective": ("takes an objective", "took an objective"),
    "building": ("takes a tower", "took a tower"),
}
SIDES = {"ours": "ally", "theirs": "enemy"}
DISTANCE = {"near": "nearby", "mid": "at mid range", "far": "from far away"}
RESPONSES = {"converged": "moved in", "held": "held ground", "left": "left", "present": "stayed nearby"}
OBJECTIVES = {"DRAGON": "dragon", "HORDE": "void grubs", "RIFTHERALD": "rift herald"}
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
    "lane_own_side": "from own side of the lane",
    "lane_middle": "from the lane's river crossing",
    "lane_enemy_side": "from the enemy side of the lane",
    "own_jungle": "from own jungle",
    "enemy_jungle": "from the enemy jungle",
    "river": "from the river",
    "other": "from near base",
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
OUTCOME_WORDS = {
    "converged": "converged",
    "held": "held ground",
    "left": "left",
    "present": "stayed present",
    "absent": "didn't go",
    "died": "died",
    "fought": "fought",
    "committed": "committed",
    "rotated": "rotated over",
    "approached": "approached",
    "lane_own_side": "own side of lane",
    "lane_middle": "river crossing",
    "lane_enemy_side": "enemy side of lane",
    "own_jungle": "own jungle",
    "enemy_jungle": "enemy jungle",
    "river": "river",
    "other": "near base",
    "by3": "by 3 minutes",
    "by5": "by 5 minutes",
    "by8": "by 8 minutes",
    "late": "after 8 minutes",
    "never": "not before 15",
    "crossed": "crossed sides",
    "stayed": "stayed one side",
    "deep_own": "deep in own half",
    "own": "own side",
    "mid": "middle",
    "theirs": "enemy side",
    "deep_theirs": "deep in enemy half",
    "off_lane": "off lane",
    "dead": "dead",
}


def outcome_words(outcome: str) -> str:
    return OUTCOME_WORDS.get(outcome, outcome.replace("_", " "))


def tendency_words(kind: str, pattern: str, half: str) -> str:
    return f"{KIND_SITUATIONS[kind]} {_priority(pattern)}, {HALVES[half]}"


ACTS = {
    "converged": ("moves in", "move in"),
    "held": ("holds ground", "hold ground"),
    "left": ("leaves", "leave"),
    "present": ("stays nearby", "stay nearby"),
    "died": ("dies", "die"),
    "fought": ("fights", "fight"),
    "committed": ("commits", "commit"),
    "rotated": ("rotates in", "rotate in"),
    "approached": ("approaches", "approach"),
    "crossed": ("crosses to the other side", "cross to the other side"),
    "stayed": ("stays on one side", "stay on one side"),
    "off_lane": ("leaves lane", "leave lane"),
    "dead": ("is dead", "be dead"),
}


def _absent(situation: str) -> tuple[str, str, str, str]:
    if situation.startswith("obj_"):
        return "does not join", "not join", "don't join", "didn't join"
    if "_ours_" in situation:
        return "does not follow", "not follow", "don't follow", "didn't follow"
    return "does not respond", "not respond", "don't respond", "didn't respond"


def outcome_label(situation: str, outcome: str) -> str:
    if outcome == "absent" and situation.startswith(("rsp_", "rspg_", "obj_")):
        return _absent(situation)[3]
    return outcome_words(outcome)


def act(situation: str, outcome: str) -> tuple[str, str, str]:
    if outcome == "absent" and situation.startswith(("rsp_", "rspg_", "obj_")):
        return _absent(situation)[:3]
    third, base = _act(situation, outcome)
    return third, base, base


def _act(situation: str, outcome: str) -> tuple[str, str]:
    if situation.startswith("ward_"):
        zone = WARD_ZONES[outcome]
        return f"wards {zone}", f"ward {zone}"
    if situation.startswith("jgl_") and situation != "jgl_sides":
        kind = situation.split("_")[1]
        if outcome == "never":
            return f"makes no {kind} before 15 minutes", f"make no {kind} before 15 minutes"
        return f"makes the first {kind} {OPENING_BANDS[outcome]}", f"make the first {kind} {OPENING_BANDS[outcome]}"
    if situation.startswith("prio_") and outcome in LANE_BANDS:
        return f"stands {LANE_BANDS[outcome]}", f"stand {LANE_BANDS[outcome]}"
    return ACTS.get(outcome, (outcome_words(outcome), outcome_words(outcome)))


def situation_lead(situation: str) -> str:
    parts = situation.split("_")
    if parts[0] == "ward":
        words = WARD_TIMES[parts[1]]
    elif parts[0] == "jgl":
        words = {"gank": "for the first gank", "invade": "for the first invade", "sides": "in the opening clear"}[parts[1]]
    elif parts[0] == "prio":
        words = "in the minute before an objective" if situation.endswith("pre_objective") else "in lane"
    else:
        words = describe_situation(situation)
    return words[0].upper() + words[1:]


def _objective(side: str, objective: str) -> str:
    if side == "none":
        return f"a {OBJECTIVES[objective]} fight nobody wins"
    words = f"{SIDES[side]} {OBJECTIVES[objective]}"
    return words if objective == "HORDE" else f"an {words}"


def _event(side: str, trigger: str, band: str, past: bool = False) -> str:
    return f"an {SIDES[side]} {TRIGGERS[trigger][past]} {DISTANCE[band]}"


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
        if outcome == "absent":
            return f"{_absent(cell)[3]} when {_event(side, trigger, band, True)}"
        return f"{RESPONSES[outcome]} after {_event(side, trigger, band, True)}"
    if prefix == "rspg" and len(parts) == 6:
        return f"{describe('_'.join(['rsp', *parts[2:]]))} when {parts[1]}"
    if prefix == "obj" and len(parts) == 5:
        _, objective, side, band, outcome = parts
        return f"{OBJECTIVE_RESPONSES[outcome]} {_objective(side, objective)} {DISTANCE[band]}"
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


PRESENT_OBJECTIVE_RESPONSES = {
    "died": "dies at",
    "fought": "fights at",
    "committed": "commits to",
    "rotated": "rotates toward",
    "approached": "approaches",
}
PRESENT_OPENINGS = {"gank": "ganks first", "invade": "invades first"}
PRESENT_JUNGLE_SIDES = {"crossed": "crosses to the other side of the jungle", "stayed": "stays on one side of the jungle"}


def phrase(cell: str) -> str:
    parts = cell.split("_")
    prefix = parts[0]
    if prefix == "rsp" and len(parts) == 5:
        _, trigger, side, band, outcome = parts
        if outcome == "absent":
            return f"{_absent(cell)[0]} when {_event(side, trigger, band)}"
        return f"{ACTS[outcome][0]} after {_event(side, trigger, band)}"
    if prefix == "rspg" and len(parts) == 6:
        return f"{phrase('_'.join(['rsp', *parts[2:]]))} when {parts[1]}"
    if prefix == "obj" and len(parts) == 5:
        _, objective, side, band, outcome = parts
        verb = "does not join" if outcome == "absent" else PRESENT_OBJECTIVE_RESPONSES[outcome]
        return f"{verb} {_objective(side, objective)} {DISTANCE[band]}"
    if prefix == "ward" and len(parts) >= 3:
        return f"places wards {WARD_TIMES[parts[1]]} {WARD_ZONES['_'.join(parts[2:])]}"
    if prefix == "jgl" and parts[1] == "sides":
        return PRESENT_JUNGLE_SIDES[parts[2]]
    if prefix == "jgl":
        return f"{PRESENT_OPENINGS[parts[1]]} {OPENING_BANDS[parts[2]]}"
    if prefix == "prio":
        situation = "pre_objective" if parts[1] == "pre" else "all"
        outcome = "_".join(parts[3:] if situation == "pre_objective" else parts[2:])
        when = PRIORITY_TIMES[situation]
        if outcome == "off_lane":
            return f"spends time away from lane{when}"
        if outcome == "dead":
            return f"spends time dead{when}"
        own, opponent, farming = _lane_outcome(outcome)
        against = "the opponent away" if opponent == "absent" else f"the opponent {LANE_BANDS[opponent]}"
        return f"{'farms' if farming == 'farm' else 'waits'} {LANE_BANDS[own]} with {against}{when}"
    if prefix == "tend" and len(parts) >= 4:
        kind = "_".join(parts[1:-2])
        words = KINDS[kind]
        if kind == "greed_punished":
            words = "gets punished for greed"
        return f"{words} {_priority(parts[-2])}, {HALVES[parts[-1]]}"
    return describe(cell)


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
    if prefix == "rspg" and len(parts) == 6:
        return "_".join(parts[:5])
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
        return f"after {_event(parts[2], parts[1], parts[3])}"
    if prefix == "rspg" and len(parts) == 5:
        return f"when {parts[1]}, {describe_situation('_'.join(['rsp', *parts[2:]]))}"
    if prefix == "obj" and len(parts) == 4:
        return f"at {_objective(parts[2], parts[1])} {DISTANCE[parts[3]]}"
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
