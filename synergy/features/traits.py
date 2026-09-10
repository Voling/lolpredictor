import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from ..config import Settings, get_settings
from .normalise import player_residuals

TRAIT_COUNT = 6
MIN_TRAIT_GAMES = 8
MIN_RELIABILITY = 0.45
TRAIT_NAMES = tuple(f"t{index + 1}" for index in range(TRAIT_COUNT))
RIDGE = 1e-6


def _split_half_reliability(scores: np.ndarray, players: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(scores)
    frame["player"] = players
    frame["half"] = rng.integers(0, 2, len(frame))
    left = frame[frame["half"] == 0].groupby("player").mean(numeric_only=True)
    right = frame[frame["half"] == 1].groupby("player").mean(numeric_only=True)
    shared = left.index.intersection(right.index)
    out = []
    for column in range(scores.shape[1]):
        a, b = left.loc[shared, column].to_numpy(), right.loc[shared, column].to_numpy()
        if len(a) < 20 or a.std() == 0 or b.std() == 0:
            out.append(0.0)
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        out.append(2 * r / (1 + r) if r > -1 else 0.0)
    return out


def _label(loadings: pd.Series) -> str:
    top = loadings.abs().nlargest(2).index
    parts = [
        ("less " if loadings[name] < 0 else "more ")
        + name.replace("e_", "").replace("w_", "").replace("_pm", "").replace("_", " ")
        for name in top
    ]
    return " and ".join(parts)


def fit_traits(
    participations: pd.DataFrame, columns: list[str], settings: Settings | None = None
) -> dict:
    settings = settings or get_settings()
    residuals = player_residuals(participations, columns)
    players = participations["puuid"].to_numpy()
    counts = pd.Series(players).value_counts()
    frequent = pd.Series(players).isin(counts[counts >= MIN_TRAIT_GAMES].index).to_numpy()
    matrix = residuals.to_numpy()[frequent]
    subject = players[frequent]
    if len(matrix) < 500 or len(set(subject)) < 50:
        return {}
    frame = pd.DataFrame(matrix)
    frame["player"] = subject
    centres = frame.groupby("player").transform("mean").to_numpy()
    within = np.cov((matrix - centres).T) + RIDGE * np.eye(matrix.shape[1])
    between = np.cov(frame.groupby("player").mean().to_numpy().T)
    values, vectors = eigh(between, within)
    vectors = vectors[:, np.argsort(values)[::-1]][:, :TRAIT_COUNT]
    reliability = _split_half_reliability(matrix @ vectors, subject)
    axes, labels, scores = {}, {}, {}
    for index, name in enumerate(TRAIT_NAMES):
        if index >= vectors.shape[1]:
            continue
        loadings = pd.Series(vectors[:, index], index=residuals.columns)
        loadings = loadings / loadings.abs().max()
        axes[name] = {k: round(float(v), 5) for k, v in loadings.items() if abs(v) >= 0.05}
        labels[name] = _label(loadings)
        scores[name] = round(float(reliability[index]), 3)
    payload = {
        "axes": axes,
        "labels": labels,
        "reliability": scores,
        "columns": list(residuals.columns),
        "players": int(len(set(subject))),
        "kept": [n for n in axes if scores[n] >= MIN_RELIABILITY],
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_dir / "traits.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_traits(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.processed_dir / "traits.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
