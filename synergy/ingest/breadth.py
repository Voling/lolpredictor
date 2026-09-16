import pandas as pd

from ..config import Settings, get_settings
from .store import Store

APEX = ("MASTER", "GRANDMASTER", "CHALLENGER")
MIN_GAMES_IN_POSITION = 10
BASE_PRIORITY = 1e6


def targets(settings: Settings | None = None, min_games: int = MIN_GAMES_IN_POSITION) -> pd.DataFrame:
    settings = settings or get_settings()
    parts = pd.read_parquet(
        settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"]
    )
    per_position = parts.groupby(["puuid", "position"]).size()
    qualified = sorted({puuid for puuid, _ in per_position[per_position >= min_games].index})
    seats = parts.groupby("puuid").size()
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute(
                "SELECT puuid, tier FROM players WHERE puuid = ANY(%s) AND tier = ANY(%s)",
                (qualified, list(APEX)),
            )
            apex = pd.DataFrame(cursor.fetchall())
    finally:
        store.close()
    if apex.empty:
        return apex
    apex["games"] = apex.puuid.map(seats).fillna(0).astype(int)
    return apex.sort_values("games").reset_index(drop=True)


def enqueue(settings: Settings | None = None, min_games: int = MIN_GAMES_IN_POSITION) -> dict:
    settings = settings or get_settings()
    wanted = targets(settings, min_games)
    store = Store(settings)
    try:
        for rank, row in enumerate(wanted.itertuples()):
            store.push_frontier(row.puuid, depth=0, priority=BASE_PRIORITY - rank, requeue=True)
        counts = store.frontier_counts()
    finally:
        store.close()
    return {
        "queued": int(len(wanted)),
        "thinnest": int(wanted.games.min()),
        "deepest": int(wanted.games.max()),
        "median_games": int(wanted.games.median()),
        "frontier": counts,
    }
