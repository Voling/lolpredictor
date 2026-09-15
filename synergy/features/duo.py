import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import Settings, get_settings

MIN_TOGETHER = 5
LP_SCALE = 100.0


def pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def teammate_pairs(participations: pd.DataFrame) -> pd.DataFrame:
    grouped = participations.groupby(["match_id", "team_id"])["puuid"].apply(list)
    rows = []
    for (match_id, team_id), members in grouped.items():
        for left, right in combinations(sorted(members), 2):
            rows.append((match_id, team_id, left, right))
    frame = pd.DataFrame(rows, columns=["match_id", "team_id", "puuid_a", "puuid_b"])
    frame["pair_key"] = frame["puuid_a"] + "|" + frame["puuid_b"]
    return frame


def consistent_duos(participations: pd.DataFrame, min_together: int = MIN_TOGETHER) -> set[str]:
    pairs = teammate_pairs(participations)
    counts = pairs.groupby("pair_key").size()
    return set(counts[counts >= min_together].index)


def _team_sides(participations: pd.DataFrame, duos: set[str]) -> pd.DataFrame:
    pairs = teammate_pairs(participations)
    flagged = pairs[pairs["pair_key"].isin(duos)]
    has = flagged.groupby(["match_id", "team_id"]).size().rename("duo_pairs").reset_index()
    sides = participations.groupby(["match_id", "team_id"]).agg(
        lp=("lp_value", "mean"), win=("win", "first")
    ).reset_index()
    sides = sides.merge(has, on=["match_id", "team_id"], how="left")
    sides["duo_pairs"] = sides["duo_pairs"].fillna(0)
    sides["duo"] = (sides["duo_pairs"] > 0).astype(float)
    other = sides.rename(columns={"team_id": "enemy_team", "lp": "enemy_lp", "duo": "enemy_duo"})
    merged = sides.merge(
        other[["match_id", "enemy_team", "enemy_lp", "enemy_duo"]], on="match_id"
    )
    return merged[merged["team_id"] != merged["enemy_team"]].dropna(subset=["lp", "enemy_lp"])


def fit_duo_effect(
    participations: pd.DataFrame,
    min_together: int = MIN_TOGETHER,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    duos = consistent_duos(participations, min_together)
    if not duos:
        return {}
    sides = _team_sides(participations, duos)
    if len(sides) < 500 or sides["duo"].nunique() < 2:
        return {}
    design = pd.DataFrame(
        {
            "lp_diff": (sides["lp"] - sides["enemy_lp"]) / LP_SCALE,
            "duo_edge": sides["duo"] - sides["enemy_duo"],
        }
    )
    target = sides["win"].astype(int).to_numpy()
    model = LogisticRegression(max_iter=2000).fit(design, target)
    lp_weight, duo_weight = float(model.coef_[0][0]), float(model.coef_[0][1])
    predicted = model.predict_proba(design)[:, 1]
    variance = predicted * (1.0 - predicted)
    information = float((design["duo_edge"] ** 2 * variance).sum())
    standard_error = float(np.sqrt(1.0 / information)) if information > 0 else float("inf")
    penalty = float(
        (sides.loc[sides["duo"] > 0, "enemy_lp"] - sides.loc[sides["duo"] > 0, "lp"]).mean()
        - (sides.loc[sides["duo"] == 0, "enemy_lp"] - sides.loc[sides["duo"] == 0, "lp"]).mean()
    )
    payload = {
        "duos": int(len(duos)),
        "team_sides": int(len(sides)),
        "min_together": int(min_together),
        "lp_weight": round(lp_weight, 5),
        "duo_weight": round(duo_weight, 5),
        "duo_sigma": round(duo_weight / standard_error, 2) if standard_error else 0.0,
        "duo_points": round(25.0 * duo_weight, 2),
        "matchmaking_penalty_lp": round(penalty, 1),
        "penalty_points": round(25.0 * lp_weight * penalty / LP_SCALE, 2),
    }
    payload["net_points"] = round(payload["duo_points"] - payload["penalty_points"], 2)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_dir / "duo_effect.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
