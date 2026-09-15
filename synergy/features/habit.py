import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .policy import ACTIONS, STATES

COMPONENTS = 10
SMOOTHING = 8.0
HABIT_COLUMNS = [f"habit_{index}" for index in range(COMPONENTS)]
TABLE = "habit.parquet"


def build_habits(settings: Settings | None = None, components: int = COMPONENTS) -> dict:
    settings = settings or get_settings()
    policy = pd.read_parquet(
        settings.processed_dir / "policy.parquet", columns=["match_id", "puuid", "state", "action"]
    )
    policy = policy[policy.state.isin(STATES) & policy.action.isin(ACTIONS)]
    cells = pd.MultiIndex.from_product([STATES, ACTIONS], names=["state", "action"])
    lookup = {pair: index for index, pair in enumerate(cells)}
    slot = policy.set_index(["state", "action"]).index.map(lookup).to_numpy()

    seats, seat_index = pd.factorize(
        policy.match_id.astype(str) + "|" + policy.puuid.astype(str)
    )
    people, person_index = pd.factorize(policy.puuid)
    width = len(cells)

    per_seat = np.zeros((len(seat_index), width))
    np.add.at(per_seat, (seats, slot), 1.0)
    per_person = np.zeros((len(person_index), width))
    np.add.at(per_person, (people, slot), 1.0)

    owner = np.zeros(len(seat_index), dtype=np.int64)
    owner[seats] = people
    others = per_person[owner] - per_seat

    base = per_person.sum(axis=0)
    base = base.reshape(len(STATES), len(ACTIONS))
    base = base / np.maximum(base.sum(axis=1, keepdims=True), 1.0)
    prior = (SMOOTHING * base).reshape(-1)

    shaped = others.reshape(-1, len(STATES), len(ACTIONS))
    totals = shaped.sum(axis=2, keepdims=True)
    rates = (shaped + prior.reshape(len(STATES), len(ACTIONS))[None]) / (totals + SMOOTHING)
    flat = np.log(rates.reshape(len(seat_index), width)) - np.log(
        np.maximum(base.reshape(-1), 1e-9)
    )[None]

    enough = others.sum(axis=1) > 0
    centre = flat[enough].mean(axis=0)
    _, _, basis = np.linalg.svd(flat[enough] - centre, full_matrices=False)
    basis = basis[:components]
    scores = np.where(enough[:, None], (flat - centre) @ basis.T, np.nan)

    keys = pd.Series(seat_index).str.split("|", n=1, expand=True)
    table = pd.DataFrame(scores, columns=HABIT_COLUMNS[:components])
    table.insert(0, "puuid", keys[1].to_numpy())
    table.insert(0, "match_id", keys[0].to_numpy())
    table.to_parquet(settings.processed_dir / TABLE, index=False)
    np.savez(settings.model_dir / "habit_basis.npz", basis=basis, centre=centre, base=base)
    return {
        "rows": int(len(table)),
        "players": int(len(person_index)),
        "cells": width,
        "components": components,
        "coverage": round(float(enough.mean()), 4),
    }


def load_habits(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *HABIT_COLUMNS])
    return pd.read_parquet(path)
