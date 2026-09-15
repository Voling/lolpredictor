import json
import logging

import pandas as pd

from ..config import Settings, get_settings
from ..features.build import load_tables
from ..features.player import _refresh_axes, build_profiles
from .dataset import PAIR_HISTORY_SOURCE, attach_dyads, build_pair_dataset, canonical_pairs
from .gold import team_advantage, team_gold
from .model import SynergyModel

logger = logging.getLogger(__name__)


def build_pair_history(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = canonical_pairs(pairs)
    available = [c for c in PAIR_HISTORY_SOURCE if c in frame.columns]
    aggregation = {
        "puuid_a": ("puuid_a", "first"),
        "puuid_b": ("puuid_b", "first"),
        "games": ("win", "count"),
        "wins": ("win", "sum"),
    }
    for column in available:
        aggregation[column] = (column, "mean")
    history = frame.groupby("pair_key").agg(**aggregation).reset_index()
    for column in PAIR_HISTORY_SOURCE:
        if column not in history.columns:
            history[column] = float("nan")
    history["winrate"] = history["wins"] / history["games"]
    return history


def train(settings: Settings | None = None, min_games: int | None = None) -> dict:
    settings = settings or get_settings()
    tables = load_tables(settings, names=("participations", "pairs", "opportunities", "dyads"))
    participations = tables["participations"]
    pairs = attach_dyads(tables["pairs"], tables.get("dyads"))
    if participations.empty or pairs.empty:
        raise RuntimeError("no processed data, run the ingest and features steps first")

    _refresh_axes(settings)
    profiles = build_profiles(
        participations, min_games=min_games, settings=settings, opportunities=tables.get("opportunities")
    )
    features, controls, _ = build_pair_dataset(
        participations, pairs, shrinkage_k=settings.style_shrinkage_k
    )

    try:
        advantage = team_advantage(settings)
    except ValueError as exc:
        logger.warning("objectives cannot be priced, training on gold at 15 alone: %s", exc)
        advantage = team_gold(settings)
    model = SynergyModel()
    report = model.fit(features, controls, advantage=advantage)

    history = build_pair_history(pairs)
    history.to_parquet(settings.processed_dir / "pair_history.parquet", index=False)
    features.to_parquet(settings.processed_dir / "pair_features.parquet", index=False)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(settings.model_dir / "synergy_model.pkl")

    payload = report.as_dict()
    payload["players_profiled"] = int(len(profiles))
    payload["pairs_with_history"] = int((history["games"] > 1).sum())
    with open(settings.model_dir / "training_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("training complete: %s", payload)
    return payload
