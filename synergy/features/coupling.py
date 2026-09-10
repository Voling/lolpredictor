import numpy as np
import pandas as pd

MIN_CELL = 30
LAG = 1


def couple(policy: pd.DataFrame, lag: int = 0, same_team: bool = True) -> pd.DataFrame:
    left = policy.copy()
    right = policy.copy()
    if lag:
        left = left.assign(minute=left["minute"] - lag)
    keys = ["match_id", "minute"]
    joined = left.merge(right, on=keys, suffixes=("_a", "_b"))
    joined = joined[joined["puuid_a"] != joined["puuid_b"]]
    if same_team:
        joined = joined[joined["team_id_a"] == joined["team_id_b"]]
    else:
        joined = joined[joined["team_id_a"] != joined["team_id_b"]]
    joined["pair_key"] = np.where(
        joined["puuid_a"] < joined["puuid_b"],
        joined["puuid_a"] + "|" + joined["puuid_b"],
        joined["puuid_b"] + "|" + joined["puuid_a"],
    )
    joined["role_pair"] = joined["role_a"] + "->" + joined["role_b"]
    return joined.reset_index(drop=True)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def conditional_mutual_information(
    frame: pd.DataFrame, a: str, b: str, given: list[str] | None = None, min_cell: int = MIN_CELL
) -> dict:
    given = given or []
    if not given:
        frame = frame.assign(_all="")
        given = ["_all"]
    total, weighted, cells = len(frame), 0.0, 0
    for _, block in frame.groupby(given, observed=True):
        if len(block) < min_cell:
            continue
        table = pd.crosstab(block[a], block[b]).to_numpy(dtype=float)
        joint = _entropy(table.ravel())
        mutual = _entropy(table.sum(axis=1)) + _entropy(table.sum(axis=0)) - joint
        weighted += mutual * len(block)
        cells += len(block)
    if cells == 0:
        return {"bits": 0.0, "coverage": 0.0, "cells": 0}
    return {"bits": weighted / cells, "coverage": cells / total, "cells": cells}


def shuffled_baseline(
    frame: pd.DataFrame, a: str, b: str, given: list[str] | None = None, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    given = given or []
    if given:
        shuffled[b] = shuffled.groupby(given, observed=True)[b].transform(
            lambda values: rng.permutation(values.to_numpy())
        )
    else:
        shuffled[b] = rng.permutation(shuffled[b].to_numpy())
    return conditional_mutual_information(shuffled, a, b, given)


def coupling_table(
    frame: pd.DataFrame, a: str = "action_a", b: str = "action_b", given: list[str] | None = None
) -> pd.DataFrame:
    rows = []
    for role_pair, block in frame.groupby("role_pair", observed=True):
        if len(block) < 500:
            continue
        observed = conditional_mutual_information(block, a, b, given)
        control = shuffled_baseline(block, a, b, given)
        rows.append(
            {
                "role_pair": role_pair,
                "rows": len(block),
                "bits": observed["bits"],
                "shuffled": control["bits"],
                "excess": observed["bits"] - control["bits"],
            }
        )
    return pd.DataFrame(rows).sort_values("excess", ascending=False)
