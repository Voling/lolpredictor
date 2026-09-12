import json

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .advantage import evaluate, load_evaluation

HORIZON = 3
PRIOR = 60.0
MIN_CELL = 40


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


def state_mix(policy: pd.DataFrame) -> pd.Series:
    return policy.groupby("state").size() / max(len(policy), 1)


def action_base(policy: pd.DataFrame) -> pd.Series:
    return policy.groupby(["state", "action"]).size() / policy.groupby("state").size()
