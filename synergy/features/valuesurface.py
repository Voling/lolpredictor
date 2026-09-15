import json
import pickle

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .advantage import evaluate, load_evaluation
from .policy import POLICY_COLUMNS

HORIZON = 3
PRIOR = 60.0
MIN_CELL = 40
CATEGORICAL = ("role_a", "role_b", "action_a", "action_b")
CONTINUOUS = [f"{column}_{side}" for column in POLICY_COLUMNS for side in ("a", "b")]
TREES = 400
LEAVES = 63
MIN_LEAF = 200


def design(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[CONTINUOUS].copy()
    for name in CATEGORICAL:
        matrix[name] = frame[name].astype("category")
    return matrix


def _swing(states: pd.DataFrame, model: dict) -> pd.DataFrame:
    board = states.copy()
    board["eval"] = evaluate(board, model)
    forward = board[["match_id", "team_id", "minute", "eval"]].copy()
    forward["minute"] -= HORIZON
    board = board.merge(
        forward.rename(columns={"eval": "later"}), on=["match_id", "team_id", "minute"]
    )
    board["swing"] = board["later"] - board["eval"]
    return board[["match_id", "team_id", "minute", "swing"]]


def _joint(policy: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    joined = policy.merge(policy, on=["match_id", "team_id", "minute"], suffixes=("_a", "_b"))
    joined = joined[joined["role_a"] < joined["role_b"]]
    return joined.merge(board, on=["match_id", "team_id", "minute"])


def fit_value_model(
    policy: pd.DataFrame, states: pd.DataFrame, settings: Settings | None = None
) -> dict:
    import lightgbm as lgb

    settings = settings or get_settings()
    model = load_evaluation(settings)
    if not model:
        return {}
    joined = _joint(policy, _swing(states, model))
    if len(joined) < 10000:
        return {}
    matches = pd.Index(sorted(joined["match_id"].unique()))
    holdout = set(matches[np.arange(len(matches)) % 5 == 0])
    fit = joined[~joined["match_id"].isin(holdout)]
    test = joined[joined["match_id"].isin(holdout)]
    booster = lgb.LGBMRegressor(
        n_estimators=TREES, learning_rate=0.05, num_leaves=LEAVES,
        min_child_samples=MIN_LEAF, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=0,
    ).fit(design(fit), fit["swing"])
    predicted = booster.predict(design(test))
    observed = test["swing"].to_numpy()
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    report = {
        "observations": int(len(joined)),
        "holdout_r2": round(1.0 - residual / total, 6) if total else 0.0,
        "trees": TREES,
        "continuous_inputs": len(CONTINUOUS),
        "horizon": HORIZON,
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_dir / "value_model.pkl", "wb") as handle:
        pickle.dump(booster, handle)
    with open(settings.processed_dir / "value_model.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def load_value_model(settings: Settings | None = None):
    settings = settings or get_settings()
    path = settings.processed_dir / "value_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def fit_value_surface(
    policy: pd.DataFrame, states: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    model = load_evaluation(settings)
    if not model:
        return pd.DataFrame()
    board = states.copy()
    board["eval"] = evaluate(board, model)
    forward = board[["match_id", "team_id", "minute", "eval"]].copy()
    forward["minute"] -= HORIZON
    board = board.merge(
        forward.rename(columns={"eval": "later"}), on=["match_id", "team_id", "minute"]
    )
    board["swing"] = board["later"] - board["eval"]
    joined = policy.merge(policy, on=["match_id", "team_id", "minute"], suffixes=("_a", "_b"))
    joined = joined[joined["role_a"] < joined["role_b"]]
    joined = joined.merge(
        board[["match_id", "team_id", "minute", "swing"]], on=["match_id", "team_id", "minute"]
    )
    grand = float(joined["swing"].mean())
    keys = ["role_a", "role_b", "state_a", "action_a", "action_b"]
    table = joined.groupby(keys, observed=True)["swing"].agg(["size", "mean"])
    table["value"] = (table["mean"] * table["size"] + grand * PRIOR) / (
        table["size"] + PRIOR
    ) - grand
    table = table[table["size"] >= MIN_CELL].reset_index()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(settings.processed_dir / "value_surface.parquet", index=False)
    with open(settings.processed_dir / "value_surface.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cells": int(len(table)),
                "observations": int(len(joined)),
                "grand_mean_swing": round(grand, 6),
                "horizon": HORIZON,
            },
            handle,
            indent=2,
        )
    return table


def load_value_surface(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / "value_surface.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
