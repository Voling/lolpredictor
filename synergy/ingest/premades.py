import pandas as pd

from ..config import Settings, get_settings
from .breadth import APEX
from .store import Store

SESSION_HOURS = 3.0
SESSION_GAMES = 2
BASE_PRIORITY = 1e7
TEAMMATE_QUERY = (
    "SELECT p.match_id, p.puuid, p.team_id, m.game_creation FROM participations p"
    " JOIN matches m USING (match_id)"
)
PLAYERS_QUERY = "SELECT puuid, tier, crawled_at FROM players WHERE puuid = ANY(%s)"
GAMES_QUERY = "SELECT puuid, count(*) AS games FROM participations WHERE puuid = ANY(%s) GROUP BY 1"


def shared_games(frame: pd.DataFrame, hours: float = SESSION_HOURS) -> pd.DataFrame:
    pairs = frame.merge(frame, on=["match_id", "team_id"])
    pairs = pairs[pairs.puuid_x < pairs.puuid_y][["puuid_x", "puuid_y", "game_creation_x"]]
    pairs = pairs.sort_values(["puuid_x", "puuid_y", "game_creation_x"])
    gaps = pairs.groupby(["puuid_x", "puuid_y"])["game_creation_x"].diff().dt.total_seconds() / 3600.0
    shared = pairs.groupby(["puuid_x", "puuid_y"]).size().rename("shared")
    close = (gaps <= hours).groupby([pairs.puuid_x, pairs.puuid_y]).sum().rename("close")
    return pd.concat([shared, close], axis=1).reset_index().rename(columns={"puuid_x": "left", "puuid_y": "right"})


def sessions(frame: pd.DataFrame, hours: float = SESSION_HOURS, needed: int = SESSION_GAMES) -> set:
    table = shared_games(frame, hours)
    return set(zip(table.left[table.close >= needed], table.right[table.close >= needed]))


def _teammates(store: Store) -> pd.DataFrame:
    with store.conn.cursor() as cursor:
        cursor.execute(TEAMMATE_QUERY)
        return pd.DataFrame(cursor.fetchall())


def premade_pairs(settings: Settings | None = None) -> set:
    store = Store(settings or get_settings())
    try:
        return sessions(_teammates(store))
    finally:
        store.close()


def seed_order(pairs: pd.DataFrame, tiers: pd.Series, games: pd.Series) -> pd.DataFrame:
    kept = pairs[tiers.reindex(pairs.left).isin(APEX).to_numpy() & tiers.reindex(pairs.right).isin(APEX).to_numpy()]
    strongest = pd.concat(
        [kept[["left", "shared"]].rename(columns={"left": "puuid"}), kept[["right", "shared"]].rename(columns={"right": "puuid"})]
    ).groupby("puuid")["shared"].max()
    order = pd.DataFrame({"strongest_pair": strongest})
    order["games"] = games.reindex(order.index).fillna(0).astype(int)
    return order.sort_values(["strongest_pair", "games"], ascending=[False, True])


def enqueue(settings: Settings | None = None, needed: int = SESSION_GAMES, limit: int | None = None) -> dict:
    settings = settings or get_settings()
    store = Store(settings)
    try:
        table = shared_games(_teammates(store))
        pairs = table[table.close >= needed]
        people = sorted(set(pairs.left) | set(pairs.right))
        with store.conn.cursor() as cursor:
            cursor.execute(PLAYERS_QUERY, (people,))
            players = pd.DataFrame(cursor.fetchall()).set_index("puuid")
            cursor.execute(GAMES_QUERY, (people,))
            games = pd.DataFrame(cursor.fetchall()).set_index("puuid")["games"]
        order = seed_order(pairs, players["tier"], games)
        if limit is not None:
            order = order.head(limit)
        for rank, row in enumerate(order.itertuples()):
            store.push_frontier(row.Index, depth=0, priority=BASE_PRIORITY + row.strongest_pair * 1e3 - rank, requeue=True)
        counts = store.frontier_counts()
    finally:
        store.close()
    both = pairs[pairs.left.isin(order.index) & pairs.right.isin(order.index)]
    return {
        "premade_pairs": int(len(pairs)),
        "queued_players": int(len(order)),
        "pairs_with_both_queued": int(len(both)),
        "shared_games_median": float(both.shared.median()) if len(both) else 0.0,
        "games_in_corpus_median": float(order.games.median()) if len(order) else 0.0,
        "frontier": counts,
    }
