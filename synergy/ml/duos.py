import json
import time

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.positions import KEY, POSITIONS
from .interaction import SEED, _basis, combination_keys
from .mirrored import SEATS_FILE, seat_gold, seats_by_position, validation_split
from .serving import CHAMPION_EFFECTS, DUO_RECORDS, DUO_REPORT, DUO_SCORES, MATRIX_FILE, PLAYER_HISTORY, SCORES_FILE, TEAM_PAIRS

KEPT_GAMES = 2
SPLIT_GAMES = 10
MIN_GROUP = 5
SCALE_BOUNDS = (0.3, 1.5)
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


def signed_residuals(gold: np.ndarray, expected: np.ndarray, blue: np.ndarray, red: np.ndarray, rows: np.ndarray) -> np.ndarray:
    signed = np.zeros(gold.shape)
    for k in range(len(POSITIONS)):
        difference = (gold[rows, blue[rows, k]] - gold[rows, red[rows, k]]) - (expected[rows, blue[rows, k]] - expected[rows, red[rows, k]])
        signed[rows, blue[rows, k]] = difference
        signed[rows, red[rows, k]] = -difference
    return signed


def joined(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(left.astype(str), "|"), right.astype(str))


def shrunk(keys: np.ndarray, values: np.ndarray, sigma2: float) -> tuple[pd.Index, np.ndarray, np.ndarray, float]:
    codes, uniques = pd.factorize(keys)
    n = np.bincount(codes, minlength=len(uniques)).astype(float)
    means = np.bincount(codes, weights=values, minlength=len(uniques)) / np.maximum(n, 1.0)
    big = n >= MIN_GROUP
    tau2 = max(float(means[big].var(ddof=1) - (sigma2 / n[big]).mean()), 0.0) if big.sum() > 2 else 0.0
    effect = means * tau2 / (tau2 + sigma2 / np.maximum(n, 1.0)) if tau2 > 0.0 else np.zeros_like(means)
    return pd.Index(uniques), effect, n, float(np.sqrt(tau2))


def lookup(index: pd.Index, effect: np.ndarray, keys: np.ndarray) -> np.ndarray:
    at = index.get_indexer(keys)
    return np.where(at >= 0, effect[np.clip(at, 0, None)], 0.0)


def r2(target: np.ndarray, guess: np.ndarray) -> float:
    return 1.0 - float(((target - guess) ** 2).sum()) / float(((target - target.mean()) ** 2).sum())


class History:
    def __init__(self, positions: np.ndarray, names: np.ndarray, champion: np.ndarray, signed: np.ndarray, rows: np.ndarray, pools: pd.DataFrame):
        seat_position = positions[rows].ravel()
        seat_name = names[rows].ravel()
        seat_champion = champion[rows].ravel()
        values = signed[rows].ravel()
        self.sigma2 = float(values.var())
        self.champion_index, self.champion_effect, self.champion_games, self.champion_tau = shrunk(joined(seat_position, seat_champion), values, self.sigma2)
        beyond = values - self.champion_played(seat_position, seat_champion)
        self.form_index, self.form_effect, self.form_games, self.form_tau = shrunk(joined(seat_name, seat_position), beyond, self.sigma2)
        expected = lookup(self.champion_index, self.champion_effect, joined(pools["position"].to_numpy(), pools["champion_name"].to_numpy()))
        weighted = pools.assign(weighted=expected * pools["games"].to_numpy())
        totals = weighted.groupby(KEY, sort=False)[["weighted", "games"]].sum()
        self.pool_index = pd.MultiIndex.from_frame(totals.reset_index()[KEY])
        self.pool_effect = (totals["weighted"] / totals["games"]).to_numpy()

    def champion_played(self, positions: np.ndarray, champions: np.ndarray) -> np.ndarray:
        return lookup(self.champion_index, self.champion_effect, joined(positions, champions))

    def champion_pool(self, names: np.ndarray, positions: np.ndarray) -> np.ndarray:
        at = self.pool_index.get_indexer(pd.MultiIndex.from_arrays([names, positions]))
        return np.where(at >= 0, self.pool_effect[np.clip(at, 0, None)], 0.0)

    def form(self, names: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return lookup(self.form_index, self.form_effect, joined(names, positions))

    def table(self) -> pd.DataFrame:
        names = [key.split("|", 1) for key in self.form_index]
        frame = pd.DataFrame({"puuid": [n for n, _ in names], "position": [p for _, p in names], "games": self.form_games.astype(np.int32), "form": self.form_effect.astype(np.float32)})
        frame["champion"] = self.champion_pool(frame["puuid"].to_numpy(), frame["position"].to_numpy()).astype(np.float32)
        return frame

    def champions(self) -> pd.DataFrame:
        parts = [key.split("|", 1) for key in self.champion_index]
        return pd.DataFrame({"position": [p for p, _ in parts], "champion": [c for _, c in parts], "games": self.champion_games.astype(np.int32), "effect": self.champion_effect.astype(np.float32)})


def champion_pools(settings: Settings) -> pd.DataFrame:
    seated = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=[*KEY, "champion_name"])
    return seated.groupby([*KEY, "champion_name"], sort=False).size().rename("games").reset_index()


def held_out(gold, predicted, blue, red, test, positions, names, champion, history: History) -> dict:
    out = {}
    for k, name in enumerate(POSITIONS):
        b, r = blue[test, k], red[test, k]
        target = gold[test, b] - gold[test, r]
        base = predicted[test, b] - predicted[test, r]
        played = history.champion_played(positions[test, b], champion[test, b]) - history.champion_played(positions[test, r], champion[test, r])
        pooled = history.champion_pool(names[test, b], positions[test, b]) - history.champion_pool(names[test, r], positions[test, r])
        form = history.form(names[test, b], positions[test, b]) - history.form(names[test, r], positions[test, r])
        served = base + pooled + form
        out[name] = {
            "readings": round(r2(target, base), 5),
            "with_champion_pool": round(r2(target, base + pooled), 5),
            "with_champion_played": round(r2(target, base + played), 5),
            "with_form": round(r2(target, served), 5),
            "slope": round(float(np.polyfit(served, target, 1)[0]), 3) if served.std() > 0 else 1.0,
        }
    return out


def fit_informative(settings: Settings) -> bool:
    path = settings.model_dir / "interaction_report.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("informative", False))


def reading_scales(checked: dict) -> np.ndarray:
    return np.clip(np.array([float(checked[name].get("slope", 1.0)) for name in POSITIONS]), *SCALE_BOUNDS)


def pair_games(
    match_id: np.ndarray, names: np.ndarray, blue: np.ndarray, red: np.ndarray, gold: np.ndarray, expected: np.ndarray, rows: np.ndarray
) -> pd.DataFrame:
    parts = []
    for side, other in ((blue, red), (red, blue)):
        for first in range(len(POSITIONS)):
            for second in range(first + 1, len(POSITIONS)):
                a, b = side[rows, first], side[rows, second]
                c, d = other[rows, first], other[rows, second]
                edge = gold[rows, a] + gold[rows, b] - gold[rows, c] - gold[rows, d]
                guess = expected[rows, a] + expected[rows, b] - expected[rows, c] - expected[rows, d]
                parts.append(
                    pd.DataFrame(
                        {
                            "match_id": match_id[rows],
                            "left": names[rows, a],
                            "right": names[rows, b],
                            "left_position": POSITIONS[first],
                            "right_position": POSITIONS[second],
                            "residual": edge - guess,
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


def served_readings(
    settings: Settings, history: pd.DataFrame | None = None, scales: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    saved = _npz(settings.model_dir / MATRIX_FILE)
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
    if history is not None and len(history):
        at = pd.MultiIndex.from_frame(history[KEY]).get_indexer(pd.MultiIndex.from_frame(styles[KEY]))
        extra = (history["form"].to_numpy(dtype=float) + history["champion"].to_numpy(dtype=float))[np.clip(at, 0, None)]
        reading = reading + np.where(at >= 0, extra, 0.0)
    if scales is not None:
        reading = reading * np.asarray(scales, dtype=float)[pd.Index(list(POSITIONS)).get_indexer(positions).clip(0)]
    return styles, vectors, reading, saved, seats


def served_totals(
    settings: Settings,
    games: pd.DataFrame,
    kept: pd.DataFrame,
    history: pd.DataFrame | None = None,
    scales: np.ndarray | None = None,
    with_fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    styles, vectors, reading, saved, _ = served_readings(settings, history, scales)
    scores = _npz(settings.model_dir / SCORES_FILE)
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
    total = reading[at_left] + reading[at_right] + (fit[inverse] if with_fit else 0.0) + record
    seat_quantiles = np.stack([np.quantile(reading[styles["position"].to_numpy() == name], GRID) if (styles["position"] == name).any() else np.zeros(len(GRID)) for name in POSITIONS])
    return total, known, seat_quantiles


def fit_duo_records(settings: Settings | None = None, stream: str = "stream.npz") -> dict:
    settings = settings or get_settings()
    clock = time.time()
    basis = _basis(settings, stream, SEED, "style")
    names, positions, match_id = basis["seat_puuid"], basis["seat_position"], basis["match_id"]
    blue, red = seats_by_position(positions, basis["seat_side"])
    gold = seat_gold(settings, match_id, names)
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    rows = np.flatnonzero(sound)
    fit = np.array(sorted(set(basis["fit"]) & set(rows)))
    test = np.array(sorted(set(basis["test"]) & set(rows)))
    predicted = seat_predictions(basis["reduced"], rows, blue, red, _npz(settings.model_dir / SEATS_FILE))
    del basis
    seated = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "champion_name"])
    lookup_champion = seated.set_index(["match_id", "puuid"])["champion_name"]
    champion = lookup_champion.reindex(pd.MultiIndex.from_arrays([np.repeat(match_id, names.shape[1]), names.ravel()])).fillna("?").to_numpy().reshape(names.shape)
    del seated, lookup_champion
    pools = champion_pools(settings)
    signed = signed_residuals(gold, predicted, blue, red, rows)
    core, check = validation_split(fit)
    validated = held_out(gold, predicted, blue, red, check, positions, names, champion, History(positions, names, champion, signed, core, pools))
    checked = held_out(gold, predicted, blue, red, test, positions, names, champion, History(positions, names, champion, signed, fit, pools))
    history = History(positions, names, champion, signed, rows, pools)
    extra = np.zeros(gold.shape)
    for k in range(len(POSITIONS)):
        for side in (blue, red):
            where = side[rows, k]
            extra[rows, where] = history.form(names[rows, where], positions[rows, where]) + history.champion_played(positions[rows, where], champion[rows, where])
    games = pair_games(match_id, names, blue, red, gold, predicted + extra, rows)
    del predicted, extra, signed
    pairs = keyed(games)
    early, late = halves(pairs)
    kept, spread, noise = duo_records(pairs, early, late)
    print(
        f"{len(rows):,} matches, {len(games):,} pair games, {len(kept):,} duos with {KEPT_GAMES}+ games together,"
        f" {spread:,.0f} gold between duos, {noise:,.0f} per game, form spread {history.form_tau:,.0f}, champion spread {history.champion_tau:,.0f}, in {time.time() - clock:.0f}s",
        flush=True,
    )
    table = history.table()
    table.to_parquet(settings.model_dir / PLAYER_HISTORY, index=False)
    history.champions().to_parquet(settings.model_dir / CHAMPION_EFFECTS, index=False)
    scales = reading_scales(validated)
    with_fit = fit_informative(settings)
    total, known, seat_quantiles = served_totals(settings, games, kept, table, scales, with_fit)
    combos = combination_keys(games.loc[known, "left_position"], games.loc[known, "right_position"])
    combo_names = sorted(set(combos))
    np.savez(
        settings.model_dir / DUO_SCORES,
        quantiles=np.quantile(total, GRID),
        combos=np.array(combo_names),
        combo_quantiles=np.stack([np.quantile(total[combos == name], GRID) for name in combo_names]),
        seat_positions=np.array(list(POSITIONS)),
        seat_quantiles=seat_quantiles,
        reading_scale=scales,
        with_fit=with_fit,
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
        "form": {"players": int(len(table)), "spread": round(history.form_tau, 1)},
        "champions": {"groups": int(len(history.champion_index)), "spread": round(history.champion_tau, 1)},
        "held_out": checked,
        "reading_scale": {name: round(float(value), 3) for name, value in zip(POSITIONS, scales)},
        "with_fit": with_fit,
        "total_gold": {f"{int(q * 100)}th": round(float(np.quantile(total, q)), 1) for q in (0.1, 0.5, 0.9)},
    }
    (settings.model_dir / DUO_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"duo scores over {int(known.sum()):,} pair games, written in {time.time() - clock:.0f}s", flush=True)
    return report
