import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .regions import REGIONS, region_of
from .timeline import VISION_WARDS, ParsedTimeline

FOLLOW_WINDOW = 2
RESPONSE_RANGE = 3000.0
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
KINDS = (
    "initiate",
    "follow",
    "defend",
    "dive",
    "fight_join",
    "overstay",
    "greed_punished",
    "ward_pre_objective",
    "ward_clear",
)
OBJECTIVE_LEAD = 2
BANKED_GOLD = 1100.0
RECALL_WINDOW = 2
PROXIMITY = 3000.0
DIVE_RADIUS = 1300.0
DIVE_REACH = 2600.0
FIGHT_RADIUS = 2000.0
TURRETS = {
    100: [(6919, 1483), (10504, 1029), (5048, 4812), (5846, 6396), (1512, 6699), (981, 10441)],
    200: [(13327, 8226), (13866, 4505), (9767, 10113), (8955, 8510), (7943, 13411), (4318, 13875)],
}
PROPENSITY_COLUMNS = [f"prop_{kind}" for kind in KINDS]
COVARIATES = ["minute", "gold_diff", "is_jungler", "in_own_half", "distance_to_enemy_jungle", "unspent_gold"]
MIN_DISPERSION = 1e-4


def _nearest_turret(x: float, y: float, owner: int) -> float:
    return min(math.hypot(x - tx, y - ty) for tx, ty in TURRETS[owner])


def _zones(parsed: ParsedTimeline, pid: int, team: int, span: int) -> list[str]:
    return [REGIONS[region_of(x, y, team)] for x, y in (parsed.positions.get(pid) or [])[: span + 1]]


def _deaths(parsed: ParsedTimeline, span: int) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {pid: set() for pid in parsed.pid_to_puuid}
    for event in parsed.kill_events():
        minute = int(float(event.get("timestamp", 0)) // 60000)
        victim = int(event.get("victimId", 0))
        if victim in out and minute <= span:
            out[victim].add(minute)
    return out


def opportunity_rows(
    match: dict, timeline: dict, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    frames = len(parsed.minutes)
    if frames < 8:
        return []
    span = frames - 1
    kills = [
        event
        for event in parsed.kill_events()
        if int(float(event.get("timestamp", 0)) // 60000) <= span
    ]
    teams = {team: [pid for pid, value in parsed.teams.items() if value == team] for team in (100, 200)}
    zones = {pid: _zones(parsed, pid, parsed.teams[pid], span) for pid in parsed.pid_to_puuid}
    deaths = _deaths(parsed, span)

    gold = {team: np.zeros(span + 1) for team in (100, 200)}
    for pid, series in parsed.gold.items():
        team = parsed.teams.get(pid, 100)
        for index in range(min(len(series), span + 1)):
            gold[team][index] += series[index]

    present: dict[int, list[set[int]]] = {}
    for team, members in teams.items():
        present[team] = [
            {pid for pid in members if minute < len(zones.get(pid, [])) and zones[pid][minute] in ENEMY_JUNGLE}
            for minute in range(span + 1)
        ]

    episodes = []
    for team in (100, 200):
        for minute in range(1, span + 1):
            if present[team][minute] and not present[team][minute - 1]:
                followers: set[int] = set()
                for step in range(minute + 1, min(minute + FOLLOW_WINDOW + 1, span + 1)):
                    followers |= present[team][step]
                episodes.append(
                    {
                        "team": team,
                        "minute": minute,
                        "initiators": set(present[team][minute]),
                        "followers": followers - present[team][minute],
                    }
                )

    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        team = parsed.teams.get(pid, 100)
        enemy_team = 200 if team == 100 else 100
        track = zones.get(pid, [])
        positions = parsed.positions.get(pid) or []
        if len(track) < 8:
            continue
        is_jungler = float(parsed.roles.get(pid, "") == "JUNGLE")
        team_episodes = [episode for episode in episodes if episode["team"] == team]
        started = {episode["minute"] for episode in team_episodes}

        def covariates(minute: int) -> dict:
            before = max(minute - 1, 0)
            zone = track[before] if before < len(track) else "UNKNOWN"
            return {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "minute": float(minute),
                "gold_diff": float(gold[team][before] - gold[enemy_team][before]) / 1000.0,
                "is_jungler": is_jungler,
                "in_own_half": float(zone in OWN_JUNGLE or zone.endswith("_OWN")),
                "distance_to_enemy_jungle": 0.0 if zone in ENEMY_JUNGLE else (
                    0.5 if zone in ("RIVER_BARON", "RIVER_DRAGON") else 1.0
                ),
                "unspent_gold": (parsed.unspent_gold.get(pid) or [0.0])[
                    min(before, len(parsed.unspent_gold.get(pid) or [0.0]) - 1)
                ] / 1000.0,
                "role": parsed.roles.get(pid, "UNKNOWN") or "UNKNOWN",
            }

        for minute in range(1, span + 1):
            if minute in deaths[pid] or (minute - 1) in deaths[pid]:
                continue
            if present[team][minute - 1]:
                continue
            row = covariates(minute)
            row["kind"] = "initiate"
            row["outcome"] = float(minute in started and pid in next(
                episode["initiators"] for episode in team_episodes if episode["minute"] == minute
            ))
            rows.append(row)

        for episode in team_episodes:
            minute = episode["minute"]
            if pid in episode["initiators"] or minute in deaths[pid]:
                continue
            row = covariates(minute)
            row["kind"] = "follow"
            row["outcome"] = float(pid in episode["followers"])
            rows.append(row)

        for minute in range(1, span + 1):
            if minute in deaths[pid]:
                continue
            invaders = [
                other
                for other in teams[enemy_team]
                if minute < len(zones.get(other, []))
                and REGIONS[region_of(*(parsed.positions.get(other) or [(0.0, 0.0)])[minute], team)] in OWN_JUNGLE
            ] if minute < len(positions) else []
            if not invaders:
                continue
            row = covariates(minute)
            row["kind"] = "defend"
            row["outcome"] = float(
                any(
                    math.hypot(
                        positions[minute][0] - (parsed.positions[other])[minute][0],
                        positions[minute][1] - (parsed.positions[other])[minute][1],
                    )
                    < RESPONSE_RANGE
                    for other in invaders
                    if minute < len(parsed.positions[other])
                )
            )
            rows.append(row)

        for minute in range(1, span + 1):
            if minute in deaths[pid] or minute >= len(positions):
                continue
            before = max(minute - 1, 0)
            if before >= len(positions):
                continue
            if _nearest_turret(*positions[before], enemy_team) > DIVE_REACH:
                continue
            row = covariates(minute)
            row["kind"] = "dive"
            row["outcome"] = float(
                any(
                    pid in parsed.involved(event)
                    and int(float(event.get("timestamp", 0)) // 60000) == minute
                    and (event.get("position") or {})
                    and _nearest_turret(
                        float(event["position"]["x"]), float(event["position"]["y"]), enemy_team
                    )
                    < DIVE_RADIUS
                    for event in kills
                )
            )
            rows.append(row)

        banked = parsed.unspent_gold.get(pid) or []
        in_base = [
            REGIONS[region_of(x, y, team)] == "BASE_OWN" for x, y in positions[: span + 1]
        ]
        for minute in range(1, span + 1):
            if minute >= len(banked) or minute >= len(in_base) or in_base[minute]:
                continue
            if banked[minute] < BANKED_GOLD:
                continue
            backs = any(
                minute + step < len(in_base) and in_base[minute + step]
                for step in range(1, RECALL_WINDOW + 1)
            )
            row = covariates(minute)
            row["kind"] = "greed_punished"
            row["outcome"] = float(any(step in deaths[pid] for step in (minute + 1, minute + 2)))
            rows.append(row)

            neighbour_backs = False
            for other in teams[team]:
                if other == pid:
                    continue
                mate = parsed.positions.get(other) or []
                if minute >= len(mate) or minute >= len(positions):
                    continue
                if math.hypot(positions[minute][0] - mate[minute][0], positions[minute][1] - mate[minute][1]) > PROXIMITY:
                    continue
                for step in range(1, RECALL_WINDOW + 1):
                    if minute + step < len(mate):
                        x, y = mate[minute + step]
                        if REGIONS[region_of(x, y, team)] == "BASE_OWN":
                            neighbour_backs = True
            if neighbour_backs:
                row = covariates(minute)
                row["kind"] = "overstay"
                row["outcome"] = float(not backs)
                rows.append(row)

        placed = {
            int(float(event.get("timestamp", 0)) // 60000)
            for event in parsed.events
            if event.get("type") == "WARD_PLACED"
            and int(event.get("creatorId", 0)) == pid
            and event.get("wardType") in VISION_WARDS
        }
        cleared = {
            int(float(event.get("timestamp", 0)) // 60000)
            for event in parsed.events
            if event.get("type") == "WARD_KILL"
            and int(event.get("killerId", 0)) == pid
            and event.get("wardType") in VISION_WARDS
        }
        contested = {
            int(float(event.get("timestamp", 0)) // 60000)
            for event in parsed.events
            if event.get("type") == "ELITE_MONSTER_KILL"
        }
        for minute in range(1, span + 1):
            if minute in deaths[pid]:
                continue
            if not any(minute + step in contested for step in range(1, OBJECTIVE_LEAD + 1)):
                continue
            row = covariates(minute)
            row["kind"] = "ward_pre_objective"
            row["outcome"] = float(minute in placed)
            rows.append(row)

        for minute in range(1, span + 1):
            if minute in deaths[pid] or minute >= len(track):
                continue
            if track[minute] not in ENEMY_JUNGLE and not track[minute].endswith("_ENEMY"):
                continue
            row = covariates(minute)
            row["kind"] = "ward_clear"
            row["outcome"] = float(minute in cleared)
            rows.append(row)

        for event in kills:
            minute = int(float(event.get("timestamp", 0)) // 60000)
            position = event.get("position") or {}
            if not position or minute > span or minute in deaths[pid]:
                continue
            if int(event.get("victimId", 0)) == pid or pid in parsed.involved(event):
                continue
            before = max(minute - 1, 0)
            if before >= len(positions):
                continue
            reach = math.hypot(
                positions[before][0] - float(position["x"]), positions[before][1] - float(position["y"])
            )
            if reach > 3 * FIGHT_RADIUS:
                continue
            row = covariates(minute)
            row["kind"] = "fight_join"
            row["outcome"] = float(
                minute < len(positions)
                and math.hypot(
                    positions[minute][0] - float(position["x"]),
                    positions[minute][1] - float(position["y"]),
                )
                < FIGHT_RADIUS
            )
            rows.append(row)
    return rows


def _gamma_prior(observed: np.ndarray, expected: np.ndarray, variance: np.ndarray) -> float:
    usable = expected > 0
    observed, expected, variance = observed[usable], expected[usable], variance[usable]
    if len(observed) < 20:
        return 1.0 / MIN_DISPERSION
    ratio = observed / expected
    weights = expected / expected.sum()
    sampling = variance / expected**2
    dispersion = float((weights * (ratio - 1.0) ** 2).sum() - (weights * sampling).sum())
    return 1.0 / max(dispersion, MIN_DISPERSION)


def fit_propensities(opportunities: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames, diagnostics = [], {}
    for kind in KINDS:
        subset = opportunities[opportunities["kind"] == kind]
        if len(subset) < 200 or subset["outcome"].nunique() < 2:
            continue
        design = pd.get_dummies(subset[[*COVARIATES, "role"]], columns=["role"], dtype=float)
        model = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000))]
        ).fit(design, subset["outcome"])
        probability = model.predict_proba(design)[:, 1]
        table = pd.DataFrame(
            {
                "puuid": subset["puuid"].to_numpy(),
                "observed": subset["outcome"].to_numpy(),
                "expected": probability,
                "variance": probability * (1.0 - probability),
            }
        )
        totals = table.groupby("puuid").agg(
            observed=("observed", "sum"),
            expected=("expected", "sum"),
            variance=("variance", "sum"),
            chances=("observed", "size"),
        )
        prior = _gamma_prior(
            totals["observed"].to_numpy(), totals["expected"].to_numpy(), totals["variance"].to_numpy()
        )
        totals[f"prop_{kind}"] = np.log(
            (totals["observed"] + prior) / (totals["expected"] + prior)
        )
        totals[f"chances_{kind}"] = totals["chances"]
        diagnostics[kind] = {
            "rows": int(len(subset)),
            "base_rate": round(float(subset["outcome"].mean()), 4),
            "player_dispersion": round(float(1.0 / prior), 5),
            "prior_strength": round(float(prior), 2),
            "median_chances": float(totals["chances"].median()),
        }
        frames.append(totals[[f"prop_{kind}", f"chances_{kind}"]])
    if not frames:
        empty = pd.DataFrame(columns=["puuid", *PROPENSITY_COLUMNS]).set_index("puuid")
        return empty, diagnostics
    out = frames[0].copy()
    for frame in frames[1:]:
        out = out.join(frame, how="outer")
    for column in PROPENSITY_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    return out.fillna(0.0), diagnostics
