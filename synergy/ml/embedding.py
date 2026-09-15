import numpy as np
import pandas as pd

from ..config import Settings, get_settings

DIMS = 64
COMPONENTS = 12
FOLDS = 5
MIN_GAMES = 5
SEED = 0
EMBED_COLUMNS = [f"style_e{index}" for index in range(COMPONENTS)]
SOURCE = "game_embeddings.parquet"
TABLE = "embedding.parquet"


def _raw(settings: Settings) -> pd.DataFrame:
    columns = ["puuid", "match_id", *[f"e{index}" for index in range(DIMS)]]
    return pd.read_parquet(settings.processed_dir / SOURCE, columns=columns)


def embedding_features(
    settings: Settings | None = None, folds: int = FOLDS, components: int = COMPONENTS
) -> dict:
    settings = settings or get_settings()
    games = _raw(settings)
    dims = [f"e{index}" for index in range(DIMS)]
    matches = games.match_id.unique()
    order = np.random.default_rng(SEED).permutation(len(matches))
    parts = np.array_split(order, folds)
    fold_of = {}
    for index, part in enumerate(parts):
        for position in part:
            fold_of[matches[position]] = index
    games["fold"] = games.match_id.map(fold_of)

    whole = games.groupby("puuid")[dims].mean()
    centre = whole.to_numpy(dtype=float).mean(axis=0)
    _, _, basis = np.linalg.svd(whole.to_numpy(dtype=float) - centre, full_matrices=False)
    basis = basis[:components]

    rows = []
    for index in range(folds):
        outside = games[games.fold != index]
        signature = outside.groupby("puuid")[dims].mean()
        counts = outside.groupby("puuid").size()
        signature = signature[counts >= MIN_GAMES]
        if signature.empty:
            continue
        scores = (signature.to_numpy(dtype=float) - centre) @ basis.T
        frame = pd.DataFrame(scores, columns=EMBED_COLUMNS[:components])
        frame["puuid"] = signature.index.to_numpy()
        frame["fold"] = index
        rows.append(frame)

    table = pd.concat(rows, ignore_index=True)
    seats = games[["match_id", "puuid"]].drop_duplicates()
    seats["fold"] = seats.match_id.map(fold_of)
    joined = seats.merge(table, on=["fold", "puuid"], how="inner").drop(columns=["fold"])
    joined.to_parquet(settings.processed_dir / TABLE, index=False)
    np.savez(settings.model_dir / "embedding_basis.npz", basis=basis, centre=centre)
    return {
        "rows": int(len(joined)),
        "players": int(joined.puuid.nunique()),
        "components": components,
        "folds": folds,
        "min_games": MIN_GAMES,
    }


def load_embedding(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *EMBED_COLUMNS])
    return pd.read_parquet(path)
