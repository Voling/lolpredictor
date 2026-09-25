import json
import time

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.positions import KEY, POSITIONS
from .interaction import SEED, _basis, combination_keys
from .mirrored import SEATS_FILE, seat_gold, seats_by_position
from .serving import DUO_RECORDS, DUO_REPORT, DUO_SCORES, MATRIX_FILE, SCORES_FILE, TEAM_PAIRS

KEPT_GAMES = 2
SPLIT_GAMES = 10
CHUNK = 65536
GRID = np.linspace(0.0, 1.0, 1001)


def _npz(path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def seat_predictions(reduced: np.ndarray, rows: np.ndarray, blue: np.ndarray, red: np.ndarray, seats: dict) -> np.ndarray:
    predicted = np.zeros(reduced.shape[:2])
    for k in range(len(POSITIONS)):
        scale = float(seats["scale"][k]) if "scale" in seats else 1.0
        for side in (blue, red):
            where = side[rows, k]
            predicted[rows, where] = scale * ((reduced[rows, where].astype(np.float64) - seats["centres"][k]) @ seats["weights"][k])
    return predicted


def pair_games(
    match_id: np.ndarray, names: np.ndarray, blue: np.ndarray, red: np.ndarray, gold: np.ndarray, predicted: np.ndarray, rows: np.ndarray
) -> pd.DataFrame:
    parts = []
    for side, other in ((blue, red), (red, blue)):
        for first in range(len(POSITIONS)):
            for second in range(first + 1, len(POSITIONS)):
                a, b = side[rows, first], side[rows, second]
                c, d = other[rows, first], other[rows, second]
                edge = gold[rows, a] + gold[rows, b] - gold[rows, c] - gold[rows, d]
                expected = predicted[rows, a] + predicted[rows, b] - predicted[rows, c] - predicted[rows, d]
                parts.append(
                    pd.DataFrame(
                        {
                            "match_id": match_id[rows],
                            "left": names[rows, a],
                            "right": names[rows, b],
                            "left_position": POSITIONS[first],
                            "right_position": POSITIONS[second],
                            "residual": edge - expected,
                        }
                    )
                )
    return pd.concat(parts, ignore_index=True)


def keyed(games: pd.DataFrame) -> pd.DataFrame:
    first = (games["left"] < games["right"]).to_numpy()
    left, right = games["left"].to_numpy(), games["right"].to_numpy()
    return pd.DataFrame(
        {"a": np.where(first, left, right), "b": np.where(first, right, left), "match_id": games["match_id"].to_numpy(), "residual": games["residual"].to_numpy()}
    )


def halves(pairs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ordered = pairs.sort_values(["a", "b", "match_id"], kind="stable")
    grouped = ordered.groupby(["a", "b"], sort=False)
    rank = grouped.cumcount().to_numpy()
    size = grouped["residual"].transform("size").to_numpy()
    long = size >= SPLIT_GAMES
    early = ordered[long & (rank < size // 2)].groupby(["a", "b"])["residual"].mean()
    late = ordered[long & (rank >= size // 2)].groupby(["a", "b"])["residual"].mean().reindex(early.index)
    return early, late


def duo_records(pairs: pd.DataFrame, early: pd.Series, late: pd.Series) -> tuple[pd.DataFrame, float, float]:
    grouped = pairs.groupby(["a", "b"], sort=False)["residual"].agg(["size", "mean"])
    noise = float(pairs["residual"].std())
    between = max(float(np.cov(early, late)[0, 1]), 0.0) if len(early) > 2 else 0.0
    grouped["record"] = grouped["mean"] * between / (between + noise**2 / grouped["size"]) if between > 0.0 else 0.0
    kept = grouped[grouped["size"] >= KEPT_GAMES].reset_index().rename(columns={"size": "games"})
    return kept, float(np.sqrt(between)), noise


def split_half(early: pd.Series, late: pd.Series) -> dict:
    if len(early) < 3:
        return {"duos": int(len(early))}
    return {
        "duos": int(len(early)),
        "r": round(float(np.corrcoef(early, late)[0, 1]), 3),
        "slope": round(float(np.polyfit(early.to_numpy(), late.to_numpy(), 1)[0]), 3),
    }


def served_totals(settings: Settings, games: pd.DataFrame, kept: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    saved = _npz(settings.model_dir / MATRIX_FILE)
    scores = _npz(settings.model_dir / SCORES_FILE)
    seats = _npz(settings.model_dir / SEATS_FILE)
    columns = [str(name) for name in saved["columns"]]
    styles = pd.read_parquet(settings.processed_dir / "player_styles.parquet", columns=[*KEY, *columns])
    vectors = styles[columns].to_numpy(dtype=np.float32)
    positions = styles["position"].to_numpy()
    reading = np.zeros(len(styles))
    for k, name in enumerate(str(value) for value in seats["positions"]):
        mask = positions == name
        scale = float(seats["scale"][k]) if "scale" in seats else 1.0
        reading[mask] = scale * ((vectors[mask].astype(np.float64) - seats["centres"][k]) @ seats["weights"][k])
    index = pd.MultiIndex.from_frame(styles[KEY])
    at_left = index.get_indexer(pd.MultiIndex.from_arrays([games["left"], games["left_position"]]))
    at_right = index.get_indexer(pd.MultiIndex.from_arrays([games["right"], games["right_position"]]))
    known = (at_left >= 0) & (at_right >= 0)
    at_left, at_right = at_left[known].astype(np.int64), at_right[known].astype(np.int64)
    unique, inverse = np.unique(at_left * len(styles) + at_right, return_inverse=True)
    matrix = saved["matrix"].astype(np.float32)
    gold_sd = float(scores["gold_sd"]) if "gold_sd" in scores else 1.0
    fit = np.empty(len(unique))
    for start in range(0, len(unique), CHUNK):
        chunk = unique[start : start + CHUNK]
        left, right = vectors[chunk // len(styles)], vectors[chunk % len(styles)]
        fit[start : start + CHUNK] = ((left @ matrix) * right).sum(axis=1).astype(np.float64) / (TEAM_PAIRS * matrix.shape[0]) * gold_sd
    pairs = keyed(games.loc[known])
    record = pairs[["a", "b"]].merge(kept[["a", "b", "record"]], on=["a", "b"], how="left")["record"].fillna(0.0).to_numpy()
    total = reading[at_left] + reading[at_right] + fit[inverse] + record
    return total, known


def fit_duo_records(settings: Settings | None = None, stream: str = "stream.npz") -> dict:
    settings = settings or get_settings()
    clock = time.time()
    basis = _basis(settings, stream, SEED, "style")
    blue, red = seats_by_position(basis["seat_position"], basis["seat_side"])
    gold = seat_gold(settings, basis["match_id"], basis["seat_puuid"])
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    rows = np.flatnonzero(sound)
    predicted = seat_predictions(basis["reduced"], rows, blue, red, _npz(settings.model_dir / SEATS_FILE))
    games = pair_games(basis["match_id"], basis["seat_puuid"], blue, red, gold, predicted, rows)
    del basis, predicted
    pairs = keyed(games)
    early, late = halves(pairs)
    kept, spread, noise = duo_records(pairs, early, late)
    print(
        f"{len(rows):,} matches, {len(games):,} pair games, {len(kept):,} duos with {KEPT_GAMES}+ games together,"
        f" {spread:,.0f} gold between duos, {noise:,.0f} per game, in {time.time() - clock:.0f}s",
        flush=True,
    )
    total, known = served_totals(settings, games, kept)
    combos = combination_keys(games.loc[known, "left_position"], games.loc[known, "right_position"])
    names = sorted(set(combos))
    np.savez(
        settings.model_dir / DUO_SCORES,
        quantiles=np.quantile(total, GRID),
        combos=np.array(names),
        combo_quantiles=np.stack([np.quantile(total[combos == name], GRID) for name in names]),
        spread=spread,
        noise=noise,
        minute=settings.target_minute,
    )
    kept[["a", "b", "games", "mean", "record"]].astype({"games": np.int32, "mean": np.float32, "record": np.float32}).to_parquet(
        settings.model_dir / DUO_RECORDS, index=False
    )
    report = {
        "minute": settings.target_minute,
        "matches": int(len(rows)),
        "pair_games": int(len(games)),
        "duos_kept": int(len(kept)),
        "spread": round(spread, 1),
        "noise": round(noise, 1),
        "split_half": split_half(early, late),
        "total_gold": {f"{int(q * 100)}th": round(float(np.quantile(total, q)), 1) for q in (0.1, 0.5, 0.9)},
    }
    (settings.model_dir / DUO_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"duo scores over {int(known.sum()):,} pair games, written in {time.time() - clock:.0f}s", flush=True)
    return report
