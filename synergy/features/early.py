import math

from .regions import MAP_SPAN, REGIONS, region_of
from .timeline import VISION_WARDS, ParsedTimeline

EARLY_MINUTES = 15
RESPONSE_RANGE = 3000.0
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
RIVER = {"RIVER_BARON", "RIVER_DRAGON"}


def _at(series: list[float], index: int) -> float:
    if not series:
        return 0.0
    return series[min(index, len(series) - 1)]


def _rate(value: float, minutes: float) -> float:
    return value / max(minutes, 1.0)


def _dealers(event: dict, key: str) -> set[int]:
    return {int(entry["participantId"]) for entry in event.get(key) or [] if entry.get("participantId")}


def early_rows(
    match: dict, timeline: dict, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    frames = len(parsed.minutes)
    if frames < 8:
        return []
    span = min(frames - 1, EARLY_MINUTES)
    minutes = float(span)
    kills = [
        event
        for event in parsed.kill_events()
        if float(event.get("timestamp", 0)) / 60000.0 <= span
    ]
    junglers = {team: parsed.role_pid(team, "JUNGLE") for team in (100, 200)}

    team_gold = {100: 0.0, 200: 0.0}
    team_damage = {100: 0.0, 200: 0.0}
    for pid, series in parsed.gold.items():
        team_gold[parsed.teams.get(pid, 100)] += _at(series, span)
    for pid, series in parsed.damage_done.items():
        team_damage[parsed.teams.get(pid, 100)] += _at(series, span)

    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        team = parsed.teams.get(pid, 100)
        enemy_team = 200 if team == 100 else 100
        positions = parsed.positions.get(pid) or []
        if len(positions) <= 1:
            continue
        window = positions[: span + 1]
        zones = [REGIONS[region_of(x, y, team)] for x, y in window]
        total = max(len(zones), 1)

        jungle_series = parsed.jungle_cs.get(pid) or []
        invade_cs = 0.0
        for index in range(1, min(len(jungle_series), span + 1)):
            if zones[index] in ENEMY_JUNGLE:
                invade_cs += max(jungle_series[index] - jungle_series[index - 1], 0.0)
        jungle_total = _at(jungle_series, span)

        invade_takedowns = 0
        ally_jungle_takedowns = 0
        focus_sizes: list[int] = []
        outnumbered: list[int] = []
        damaged_enemies: set[int] = set()
        solo_involvements = 0
        involvements = 0
        for event in kills:
            position = event.get("position") or {}
            involved = parsed.involved(event)
            if pid in involved:
                involvements += 1
                zone = REGIONS[region_of(float(position.get("x", 0.0)), float(position.get("y", 0.0)), team)]
                invade_takedowns += zone in ENEMY_JUNGLE
                ally_jungle_takedowns += zone in OWN_JUNGLE
                allies = {
                    dealer
                    for dealer in _dealers(event, "victimDamageReceived")
                    if parsed.teams.get(dealer) == team
                }
                focus_sizes.append(max(len(allies), 1))
                solo_involvements += len(allies) <= 1
            if pid in _dealers(event, "victimDamageReceived"):
                victim = int(event.get("victimId", 0))
                if parsed.teams.get(victim) == enemy_team:
                    damaged_enemies.add(victim)
            if int(event.get("victimId", 0)) == pid:
                attackers = {
                    dealer
                    for dealer in _dealers(event, "victimDamageReceived")
                    if parsed.teams.get(dealer) == enemy_team
                }
                outnumbered.append(max(len(attackers), 1))

        own_jungler = junglers.get(team)
        enemy_jungler = junglers.get(enemy_team)
        proximity = _mean_proximity(parsed, pid, own_jungler, span)
        enemy_proximity = _mean_proximity(parsed, pid, enemy_jungler, span)
        invasions, responses = 0, 0
        if enemy_jungler is not None and enemy_jungler != pid:
            enemy_track = parsed.positions.get(enemy_jungler) or []
            for index in range(min(len(enemy_track), len(window))):
                x, y = enemy_track[index]
                if REGIONS[region_of(x, y, team)] not in OWN_JUNGLE:
                    continue
                invasions += 1
                if math.hypot(window[index][0] - x, window[index][1] - y) < RESPONSE_RANGE:
                    responses += 1

        wards = control_wards = wards_killed = 0
        purchases: list[float] = []
        for event in parsed.events:
            minute = float(event.get("timestamp", 0)) / 60000.0
            if minute > span:
                continue
            kind = event.get("type")
            if kind == "WARD_PLACED" and int(event.get("creatorId", 0)) == pid:
                if event.get("wardType") in VISION_WARDS:
                    wards += 1
                    control_wards += event.get("wardType") == "CONTROL_WARD"
            elif kind == "WARD_KILL" and int(event.get("killerId", 0)) == pid:
                wards_killed += event.get("wardType") in VISION_WARDS
            elif kind == "ITEM_PURCHASED" and int(event.get("participantId", 0)) == pid:
                purchases.append(minute)

        damage_done = _at(parsed.damage_done.get(pid) or [], span)
        damage_taken = _at(parsed.damage_taken.get(pid) or [], span)
        exchange = damage_done + damage_taken
        gold = _at(parsed.gold.get(pid) or [], span)
        first_back = next((minute for minute in purchases if minute >= 1.0), float(span))

        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "e_enemy_jungle_share": sum(zone in ENEMY_JUNGLE for zone in zones) / total,
                "e_own_jungle_share": sum(zone in OWN_JUNGLE for zone in zones) / total,
                "e_river_share": sum(zone in RIVER for zone in zones) / total,
                "e_lane_share": sum(zone.startswith("LANE_") for zone in zones) / total,
                "e_enemy_half_share": sum(zone.endswith("_ENEMY") for zone in zones) / total,
                "e_invade_cs": invade_cs,
                "e_invade_cs_share": invade_cs / jungle_total if jungle_total > 0 else 0.0,
                "e_invade_takedowns_pm": _rate(invade_takedowns, minutes),
                "e_ally_jungle_takedowns_pm": _rate(ally_jungle_takedowns, minutes),
                "e_counter_invade_share": responses / invasions if invasions else 0.0,
                "e_jungler_proximity": proximity,
                "e_enemy_jungler_proximity": enemy_proximity,
                "e_damage_done_pm": _rate(damage_done, minutes),
                "e_damage_taken_pm": _rate(damage_taken, minutes),
                "e_trade_ratio": damage_done / exchange if exchange > 0 else 0.5,
                "e_distinct_enemies_damaged": float(len(damaged_enemies)),
                "e_focus_fire": sum(focus_sizes) / len(focus_sizes) if focus_sizes else 0.0,
                "e_solo_involvement_share": solo_involvements / involvements if involvements else 0.0,
                "e_outnumbered_on_death": sum(outnumbered) / len(outnumbered) if outnumbered else 0.0,
                "e_wards_pm": _rate(wards, minutes),
                "e_control_wards_pm": _rate(control_wards, minutes),
                "e_wards_killed_pm": _rate(wards_killed, minutes),
                "e_first_back_minute": first_back,
                "e_gold_at_15": gold,
                "e_xp_at_15": _at(parsed.xp.get(pid) or [], span),
                "e_cs_at_15": _at(parsed.cs.get(pid) or [], span) - jungle_total,
                "e_jungle_cs_at_15": jungle_total,
                "e_level_at_15": _at(parsed.levels.get(pid) or [], span),
                "e_gold_share": gold / team_gold[team] if team_gold[team] > 0 else 0.0,
                "e_damage_share": damage_done / team_damage[team] if team_damage[team] > 0 else 0.0,
            }
        )
    return _add_lane_differentials(rows, match)


def _mean_proximity(parsed: ParsedTimeline, pid: int, other: int | None, span: int) -> float:
    if other is None or other == pid:
        return 0.0
    left = parsed.positions.get(pid) or []
    right = parsed.positions.get(other) or []
    steps = min(len(left), len(right), span + 1)
    if steps == 0:
        return 0.0
    distance = sum(math.hypot(left[i][0] - right[i][0], left[i][1] - right[i][1]) for i in range(steps))
    return 1.0 - (distance / steps) / MAP_SPAN


LANE_OUTCOMES = ("e_gold_at_15", "e_xp_at_15", "e_cs_at_15")


def _add_lane_differentials(rows: list[dict], match: dict) -> list[dict]:
    role = {p["puuid"]: (p.get("teamPosition") or p.get("individualPosition") or "").upper()
            for p in match["info"]["participants"]}
    team = {p["puuid"]: int(p.get("teamId", 0)) for p in match["info"]["participants"]}
    by_role: dict[str, list[dict]] = {}
    for row in rows:
        position = role.get(row["puuid"], "")
        if position:
            by_role.setdefault(position, []).append(row)
    for position, members in by_role.items():
        if len(members) != 2 or team.get(members[0]["puuid"]) == team.get(members[1]["puuid"]):
            continue
        left, right = members
        for column in LANE_OUTCOMES:
            a, b = left.get(column), right.get(column)
            if a is None or b is None:
                continue
            left[f"{column}_diff"] = float(a) - float(b)
            right[f"{column}_diff"] = float(b) - float(a)
    for row in rows:
        for column in LANE_OUTCOMES:
            row.setdefault(f"{column}_diff", 0.0)
    return rows


EARLY_FEATURE_COLUMNS = [
    "e_enemy_jungle_share",
    "e_own_jungle_share",
    "e_river_share",
    "e_lane_share",
    "e_enemy_half_share",
    "e_invade_cs",
    "e_invade_cs_share",
    "e_invade_takedowns_pm",
    "e_ally_jungle_takedowns_pm",
    "e_counter_invade_share",
    "e_jungler_proximity",
    "e_enemy_jungler_proximity",
    "e_damage_done_pm",
    "e_damage_taken_pm",
    "e_trade_ratio",
    "e_distinct_enemies_damaged",
    "e_focus_fire",
    "e_solo_involvement_share",
    "e_outnumbered_on_death",
    "e_wards_pm",
    "e_control_wards_pm",
    "e_wards_killed_pm",
    "e_first_back_minute",
    "e_gold_at_15",
    "e_gold_at_15_diff",
    "e_xp_at_15_diff",
    "e_cs_at_15_diff",
    "e_xp_at_15",
    "e_cs_at_15",
    "e_jungle_cs_at_15",
    "e_level_at_15",
    "e_gold_share",
    "e_damage_share",
]
