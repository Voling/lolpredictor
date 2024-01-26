import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .early import EARLY_FEATURE_COLUMNS
from .player import STYLE_AXES, _style_scores

MIN_CHAMPION_GAMES = 15
CHAMPION_SHRINKAGE = 20.0
STYLE_COLUMNS = [f"style_{name}" for name in STYLE_AXES]


def role_normaliser(frame: pd.DataFrame) -> dict:
    columns = [column for column in EARLY_FEATURE_COLUMNS if column in frame.columns]
    grouped = frame.groupby("position")[columns]
    return {
        "columns": columns,
        "mean": grouped.mean().to_dict(orient="index"),
        "std": grouped.std().replace(0, np.nan).fillna(1.0).to_dict(orient="index"),
        "global_mean": frame[columns].mean().to_dict(),
        "global_std": frame[columns].std().replace(0, np.nan).fillna(1.0).to_dict(),
    }


def apply_role_normaliser(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    columns = [column for column in stats["columns"] if column in frame.columns]
    roles = frame["position"].astype(str).to_numpy()
    centre = pd.DataFrame(
        [stats["mean"].get(role, stats["global_mean"]) for role in roles], index=frame.index
    )[columns]
    spread = pd.DataFrame(
        [stats["std"].get(role, stats["global_std"]) for role in roles], index=frame.index
    )[columns].replace(0, 1.0)
    scaled = (frame[columns].astype(float) - centre) / spread
    return scaled.clip(-4, 4).fillna(0.0)


def champion_profiles(
    participations: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, dict]:
    settings = settings or get_settings()
    stats = role_normaliser(participations)
    scaled = apply_role_normaliser(participations, stats)
    styles = _style_scores(scaled)
    frame = pd.concat(
        [participations[["champion_name", "position", "win"]].reset_index(drop=True),
         styles.reset_index(drop=True)],
        axis=1,
    )
    grouped = frame.groupby(["champion_name", "position"])
    profile = grouped.agg(games=("win", "size"), winrate=("win", "mean"))
    profile = profile.join(grouped[STYLE_COLUMNS].mean())
    weight = profile["games"] / (profile["games"] + CHAMPION_SHRINKAGE)
    profile[STYLE_COLUMNS] = profile[STYLE_COLUMNS].mul(weight, axis=0)
    profile["confidence"] = weight.round(3)
    profile = profile[profile["games"] >= MIN_CHAMPION_GAMES].reset_index()
    for column in STYLE_COLUMNS:
        profile[f"{column}_pct"] = (profile[column].rank(pct=True) * 100).round(1)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(settings.processed_dir / "champion_profiles.parquet", index=False)
    return profile, stats
