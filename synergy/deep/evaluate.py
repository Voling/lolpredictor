import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings, get_settings
from ..features.build import load_tables
from ..ml.dataset import build_pair_dataset, build_team_dataset
from ..ml.model import OBSERVED_LINEUP, RIDGE_ALPHAS

logger = logging.getLogger(__name__)


def _leave_one_out_embeddings(games: pd.DataFrame, columns: list[str], shrinkage: float) -> pd.DataFrame:
    grouped = games.groupby("puuid")
    counts = (grouped["puuid"].transform("count") - 1).to_numpy(dtype=float)
    totals = grouped[columns].transform("sum").to_numpy()
    excluded = totals - games[columns].to_numpy()
    means = np.divide(excluded, counts[:, None], out=np.zeros_like(excluded), where=counts[:, None] > 0)
    weight = np.where(counts > 0, counts / (counts + shrinkage), 0.0)
    out = pd.DataFrame(means * weight[:, None], columns=columns, index=games.index)
    out["match_id"] = games["match_id"].to_numpy()
    out["puuid"] = games["puuid"].to_numpy()
    return out


def _cross_validated_auc(X: pd.DataFrame, y: np.ndarray, phi: list[str], controls: list[str]):
    baseline = np.zeros(len(X))
    combined = np.zeros(len(X))
    for train, test in KFold(5, shuffle=True, random_state=42).split(X):
        model = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=2000))]
        ).fit(X[controls].iloc[train], y[train])
        baseline[test] = model.predict_proba(X[controls].iloc[test])[:, 1]
        residual = Pipeline(
            [("scale", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))]
        ).fit(X[phi].iloc[train], y[train] - model.predict_proba(X[controls].iloc[train])[:, 1])
        combined[test] = baseline[test] + residual.predict(X[phi].iloc[test])
    return baseline, np.clip(combined, 1e-4, 1 - 1e-4)


def compare_representations(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    tables = load_tables(settings)
    participations, pairs = tables["participations"], tables["pairs"]

    path = settings.processed_dir / "game_embeddings.parquet"
    if not path.exists():
        raise RuntimeError("game embeddings missing, run `python -m synergy deep-train` first")
    games = pd.read_parquet(path)
    columns = [column for column in games.columns if column.startswith("e")]
    loo = _leave_one_out_embeddings(games, columns, settings.style_shrinkage_k)
    keyed = loo.set_index(["match_id", "puuid"])

    features, controls, _ = build_pair_dataset(
        participations, pairs, shrinkage_k=settings.style_shrinkage_k
    )
    handwritten_X, y, observed = build_team_dataset(features, controls)
    stratum = observed >= OBSERVED_LINEUP

    left = keyed[columns].reindex(
        pd.MultiIndex.from_arrays([features["match_id"], features["puuid_a"]])
    ).to_numpy()
    right = keyed[columns].reindex(
        pd.MultiIndex.from_arrays([features["match_id"], features["puuid_b"]])
    ).to_numpy()
    left, right = np.nan_to_num(left), np.nan_to_num(right)
    product = pd.DataFrame(left * right, columns=[f"prod_{c}" for c in columns])
    gap = pd.DataFrame(np.abs(left - right), columns=[f"gap_{c}" for c in columns])
    deep_phi = pd.concat([product, gap], axis=1)
    deep_phi["match_id"] = features["match_id"].to_numpy()
    deep_phi["team_id"] = features["team_id"].to_numpy()
    deep_phi["win"] = features["win"].to_numpy()

    per_team = keyed.reset_index().merge(
        participations[["match_id", "puuid", "team_id"]], on=["match_id", "puuid"], how="inner"
    )
    team_sums = per_team.groupby(["match_id", "team_id"])[columns].sum()
    team_sums.columns = [f"team_{c}" for c in columns]
    deep_controls = controls.set_index(["match_id", "team_id"]).join(team_sums).reset_index()

    phi_columns = list(product.columns) + list(gap.columns)
    grouped = deep_phi.groupby(["match_id", "team_id"])
    merged = grouped[phi_columns].sum().join(grouped["win"].first())
    merged = merged.join(deep_controls.set_index(["match_id", "team_id"])).dropna()
    blue, red = merged.xs(100, level="team_id"), merged.xs(200, level="team_id")
    shared = blue.index.intersection(red.index)
    blue, red = blue.loc[shared], red.loc[shared]
    control_columns = [f"team_{c}" for c in columns] + [
        "team_skill",
        "team_lp",
        "team_lp_coverage",
        "team_experience",
        "team_season_winrate",
        "team_season_coverage",
    ]
    deep_X = blue[phi_columns + control_columns] - red[phi_columns + control_columns]
    deep_y = blue["win"].to_numpy().astype(int)

    handwritten_columns = [c for c in handwritten_X.columns if not c.startswith("team_")]
    hand_controls = [c for c in handwritten_X.columns if c.startswith("team_")]
    hand_base, hand_combined = _cross_validated_auc(
        handwritten_X, y, handwritten_columns, hand_controls
    )
    deep_base, deep_combined = _cross_validated_auc(deep_X, deep_y, phi_columns, control_columns)

    deep_stratum = pd.Series(stratum, index=handwritten_X.index).reindex(deep_X.index).fillna(False)
    deep_stratum = deep_stratum.to_numpy().astype(bool)

    payload = {
        "matches": int(len(handwritten_X)),
        "deep_matches": int(len(deep_X)),
        "well_observed_matches": int(stratum.sum()),
        "embedding_dimensions": len(columns),
        "handwritten": {
            "auc_all": round(float(roc_auc_score(y, hand_combined)), 4),
            "baseline_auc_all": round(float(roc_auc_score(y, hand_base)), 4),
            "auc_well_observed": round(float(roc_auc_score(y[stratum], hand_combined[stratum])), 4),
            "baseline_auc_well_observed": round(float(roc_auc_score(y[stratum], hand_base[stratum])), 4),
        },
        "learned": {
            "auc_all": round(float(roc_auc_score(deep_y, deep_combined)), 4),
            "baseline_auc_all": round(float(roc_auc_score(deep_y, deep_base)), 4),
            "auc_well_observed": round(
                float(roc_auc_score(deep_y[deep_stratum], deep_combined[deep_stratum])), 4
            ),
            "baseline_auc_well_observed": round(
                float(roc_auc_score(deep_y[deep_stratum], deep_base[deep_stratum])), 4
            ),
        },
    }
    with open(settings.model_dir / "representation_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("representation comparison: %s", payload)
    return payload
