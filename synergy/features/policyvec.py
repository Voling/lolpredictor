import json

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .policy import ACTIONS, STATES

MIN_MINUTES = 40
MIN_STATE_ROWS = 2000
MIN_PLAYERS = 150
MIN_RELIABILITY = 0.50


def _cells(policy: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for state in STATES:
        block = policy[policy["state"] == state]
        if len(block) < MIN_STATE_ROWS:
            continue
        for action in ACTIONS:
            out.append((state, action))
    return out


def _reliability(policy: pd.DataFrame, cells: list[tuple[str, str]], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    frame = policy.copy()
    frame["_half"] = rng.integers(0, 2, len(frame))
    counts = frame["puuid"].value_counts()
    frame = frame[frame["puuid"].isin(counts[counts >= MIN_MINUTES].index)]
    scores = {}
    for state, action in cells:
        block = frame[frame["state"] == state]
        if block.empty:
            continue
        block = block.assign(value=(block["action"] == action).astype(float))
        left = block[block["_half"] == 0].groupby("puuid")["value"].mean()
        right = block[block["_half"] == 1].groupby("puuid")["value"].mean()
        shared = left.index.intersection(right.index)
        if len(shared) < MIN_PLAYERS or left.loc[shared].std() == 0 or right.loc[shared].std() == 0:
            continue
        r = float(np.corrcoef(left.loc[shared], right.loc[shared])[0, 1])
        if r > -1:
            scores[f"{state}|{action}"] = round(2 * r / (1 + r), 4)
    return scores


def fit_policy_vectors(
    policy: pd.DataFrame, settings: Settings | None = None, min_reliability: float = MIN_RELIABILITY
) -> tuple[pd.DataFrame, dict]:
    settings = settings or get_settings()
    cells = _cells(policy)
    scores = _reliability(policy, cells)
    kept = [name for name, value in scores.items() if value >= min_reliability]
    if not kept:
        return pd.DataFrame(), {}
    counts = policy["puuid"].value_counts()
    frequent = policy[policy["puuid"].isin(counts[counts >= MIN_MINUTES].index)]
    columns = {}
    for name in kept:
        state, action = name.split("|")
        block = frequent[frequent["state"] == state]
        columns[name] = block.assign(value=(block["action"] == action).astype(float)).groupby(
            "puuid"
        )["value"].mean()
    vectors = pd.DataFrame(columns)
    vectors = vectors.fillna(vectors.mean())
    report = {
        "cells_tested": len(scores),
        "cells_kept": len(kept),
        "min_reliability": min_reliability,
        "players": int(len(vectors)),
        "reliability": {name: scores[name] for name in kept},
        "mean": {name: round(float(vectors[name].mean()), 5) for name in kept},
        "std": {name: round(float(vectors[name].std() or 1.0), 5) for name in kept},
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    vectors.reset_index().rename(columns={"index": "puuid"}).to_parquet(
        settings.processed_dir / "policy_vectors.parquet", index=False
    )
    with open(settings.processed_dir / "policy_vectors.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return vectors, report


def load_policy_vectors(settings: Settings | None = None) -> tuple[pd.DataFrame, dict]:
    settings = settings or get_settings()
    path = settings.processed_dir / "policy_vectors.parquet"
    meta = settings.processed_dir / "policy_vectors.json"
    if not path.exists() or not meta.exists():
        return pd.DataFrame(), {}
    vectors = pd.read_parquet(path).set_index("puuid")
    with open(meta, encoding="utf-8") as handle:
        return vectors, json.load(handle)


def standardise(vectors: pd.DataFrame, report: dict, weighted: bool = True) -> np.ndarray:
    columns = list(report["reliability"])
    centre = np.array([report["mean"][name] for name in columns])
    spread = np.array([report["std"][name] or 1.0 for name in columns])
    matrix = (vectors[columns].to_numpy(dtype=float) - centre) / spread
    if weighted:
        matrix = matrix * np.sqrt(
            np.clip([report["reliability"][name] for name in columns], 0.0, None)
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def similarity(vectors: pd.DataFrame, report: dict) -> tuple[np.ndarray, dict]:
    unit = standardise(vectors, report)
    return unit, {puuid: index for index, puuid in enumerate(vectors.index)}


def neighbours(
    puuid: str,
    vectors: pd.DataFrame,
    report: dict,
    limit: int = 10,
    roles: pd.Series | None = None,
    same_role: bool = True,
) -> pd.Series:
    unit, index = similarity(vectors, report)
    if puuid not in index:
        return pd.Series(dtype=float)
    scores = pd.Series(unit @ unit[index[puuid]], index=vectors.index).drop(puuid)
    if same_role and roles is not None and puuid in roles.index:
        mine = roles.loc[puuid]
        eligible = roles.reindex(scores.index)
        scores = scores[eligible == mine]
    return scores.nlargest(limit)
