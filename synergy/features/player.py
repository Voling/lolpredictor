import json
import logging

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .early import EARLY_FEATURE_COLUMNS
from .wave import WAVE_FEATURE_COLUMNS
from .normalise import apply_normaliser, fit_normaliser, player_residuals
from .propensity import PROPENSITY_COLUMNS, fit_propensities

logger = logging.getLogger(__name__)

STYLE_AXES: dict[str, dict[str, float]] = {
    "invading": {
        "e_enemy_jungle_share": 1.0,
        "e_invade_cs_share": 0.9,
        "e_enemy_half_share": 0.5,
    },
    "aggression": {
        "e_damage_done_pm": 1.0,
        "e_trade_ratio": 0.8,
        "e_fight_volume_pm": 0.7,
        "e_damage_share": 0.6,
    },
    "frontline": {
        "e_damage_taken_pm": 1.0,
        "e_trade_ratio": -0.7,
    },
    "vision": {
        "e_wards_pm": 1.0,
        "e_control_wards_pm": 0.9,
        "e_wards_killed_pm": 0.6,
    },
    "farming": {
        "e_cs_at_15": 1.0,
        "e_gold_at_15": 0.6,
        "e_jungle_cs_at_15": 0.5,
        "e_xp_at_15": 0.5,
    },
    "tempo": {
        "e_first_back_minute": -1.0,
        "e_level_at_15": 0.8,
        "e_gold_at_15": 0.5,
    },
    "lane_control": {
        "w_push_share": 1.0,
        "w_first_push_minute": -0.7,
    },
}

STYLE_COLUMNS = [f"style_{name}" for name in STYLE_AXES]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in EARLY_FEATURE_COLUMNS + WAVE_FEATURE_COLUMNS if column in frame.columns]


def normaliser(frame: pd.DataFrame) -> dict:
    return fit_normaliser(frame, feature_columns(frame))


def _style_scores(scaled: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=scaled.index)
    for axis, weights in STYLE_AXES.items():
        usable = {column: weight for column, weight in weights.items() if column in scaled.columns}
        if not usable:
            out[f"style_{axis}"] = 0.0
            continue
        total = sum(abs(weight) for weight in usable.values())
        out[f"style_{axis}"] = sum(scaled[column] * weight for column, weight in usable.items()) / total
    return out


def participation_styles(participations: pd.DataFrame, stats: dict) -> pd.DataFrame:
    scaled = apply_normaliser(participations, stats)
    styles = _style_scores(scaled)
    keys = participations[["match_id", "puuid", "team_id", "position", "win"]].reset_index(drop=True)
    return pd.concat([keys, styles.reset_index(drop=True)], axis=1)


def load_normaliser(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    with open(settings.processed_dir / "normaliser.json", encoding="utf-8") as handle:
        return json.load(handle)


def build_profiles(
    participations: pd.DataFrame,
    min_games: int | None = None,
    settings: Settings | None = None,
    opportunities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    settings = settings or get_settings()
    min_games = settings.min_profile_games if min_games is None else min_games
    stats = normaliser(participations)
    scaled = player_residuals(participations, stats["columns"])
    styles = _style_scores(scaled)
    enriched = pd.concat(
        [participations.reset_index(drop=True), scaled.reset_index(drop=True), styles.reset_index(drop=True)],
        axis=1,
    )
    enriched = enriched.loc[:, ~enriched.columns.duplicated(keep="last")]

    grouped = enriched.groupby("puuid")
    profile = grouped.agg(games=("match_id", "count"), winrate=("win", "mean"))

    numeric_columns = stats["columns"] + STYLE_COLUMNS
    profile = profile.join(grouped[numeric_columns].mean())
    profile = profile.join(grouped[STYLE_COLUMNS].std().add_suffix("_var"))
    confidence = profile["games"] / (profile["games"] + settings.style_shrinkage_k)
    profile[STYLE_COLUMNS] = profile[STYLE_COLUMNS].mul(confidence, axis=0)
    profile["style_confidence"] = confidence.round(3)

    identity = grouped.agg(
        main_position=("position", lambda s: s.value_counts().idxmax()),
        champion_pool=("champion_name", "nunique"),
    )
    for column in ("game_name", "tag_line", "tier", "division"):
        identity[column] = grouped[column].last() if column in enriched.columns else None
    identity["lp_value"] = grouped["lp_value"].mean() if "lp_value" in enriched.columns else np.nan
    profile = profile.join(identity)
    if opportunities is not None and not opportunities.empty:
        tendencies, diagnostics = fit_propensities(opportunities)
        profile = profile.join(tendencies)
        with open(settings.processed_dir / "propensity_report.json", "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)
    for column in PROPENSITY_COLUMNS:
        if column not in profile.columns:
            profile[column] = 0.0
    profile[PROPENSITY_COLUMNS] = profile[PROPENSITY_COLUMNS].fillna(0.0)
    profile = profile[profile["games"] >= min_games].copy()

    for column in STYLE_COLUMNS + PROPENSITY_COLUMNS:
        profile[f"{column}_pct"] = (profile[column].rank(pct=True) * 100).round(1)

    profile = profile.reset_index()

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(settings.processed_dir / "player_profiles.parquet", index=False)
    with open(settings.processed_dir / "normaliser.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle)
    logger.info("built %s player profiles", len(profile))
    return profile


def load_profiles(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return pd.read_parquet(settings.processed_dir / "player_profiles.parquet")
