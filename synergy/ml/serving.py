import json
import threading
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.describe import describe, describe_situation, named, phrase, situation_of
from ..features.exposure import EXPOSURE_TABLE
from ..features.hinge import RESPONSES as HINGE_RESPONSES
from ..features.hinge import TABLE as HINGE_TABLE
from ..features.positions import POSITIONS, combination, spoken

TEAM_SIZE = 5
TEAM_PAIRS = TEAM_SIZE * (TEAM_SIZE - 1) // 2
DRIVERS = 6
DISTINCTIVE = 5
SCORE_SPAN = 6.0
SIGNIFICANT = 4
NAMES_TABLE = "player_names.parquet"
MATRIX_FILE = "interaction_matrix.npz"
SCORES_FILE = "interaction_scores.npz"
SEATS_FILE = "seat_weights.npz"
REPORT_FILE = "interaction_report.json"
DUO_RECORDS = "duo_records.parquet"
DUO_SCORES = "duo_scores.npz"
DUO_REPORT = "duo_report.json"
PLAYER_HISTORY = "player_history.parquet"
VECTORS_FILE = "style_vectors.npz"
STYLES_TABLE = "player_styles.parquet"
PROFILES_TABLE = "player_profiles.parquet"
GRID = 1001
MIN_GRID = 20
CHAMPION_EFFECTS = "champion_effects.parquet"
UNRELIABLE = "the fit did not beat shuffled partners on this corpus, so it is shown for inspection only"

_held: dict[tuple[str, str], tuple[tuple[int, int], object]] = {}
_lock = threading.RLock()


def cached(path: Path, kind: str, load):
    key = (str(path), kind)
    with _lock:
        try:
            stat = path.stat()
        except FileNotFoundError:
            _held.pop(key, None)
            return None
        version = (stat.st_mtime_ns, stat.st_size)
        held = _held.get(key)
        if held is None or held[0] != version:
            held = (version, load(path))
            _held[key] = held
        return held[1]


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _indexed_hinge(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).set_index(["responder", "actor"]).sort_index()


def _name_index(path: Path) -> dict[tuple[str, str], str]:
    table = pd.read_parquet(path, columns=["puuid", "game_name", "tag_line", "current"])
    table = table.sort_values("current", kind="stable")
    return {
        (str(name).lower(), str(tag).lower()): puuid
        for puuid, name, tag in zip(table["puuid"], table["game_name"], table["tag_line"])
    }


def _grid(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=0)
    picks = np.linspace(0, len(ordered) - 1, min(len(ordered), GRID)).round().astype(int)
    return np.ascontiguousarray(ordered[picks].T)


def percentile_among(grid: np.ndarray, value: float) -> float:
    return round(float(np.searchsorted(grid, value, side="left") / len(grid) * 100.0), 1)


def _pack(columns, puuid, position, seats, evidence, matrix, grid, exposure=None, exposure_columns=()) -> dict:
    return {
        "columns": list(columns),
        "puuid": puuid,
        "position": position,
        "at": {key: index for index, key in enumerate(zip(puuid, position))},
        "seats": seats,
        "evidence": evidence,
        "matrix": matrix,
        "grid": grid,
        "exposure": exposure,
        "exposure_columns": [str(name) for name in exposure_columns],
        "extra": {},
        "extra_history": {},
        "extra_exposure": {},
    }


def seat_of(vectors: dict, puuid: str, position: str) -> tuple[np.ndarray, int, float] | None:
    found = vectors["extra"].get((puuid, position))
    if found is not None:
        return np.asarray(found[0], dtype=float), int(found[1]), float(found[2])
    row = vectors["at"].get((puuid, position))
    if row is None:
        return None
    return vectors["matrix"][row].astype(float), int(vectors["seats"][row]), float(vectors["evidence"][row])


def exposure_of(vectors: dict, puuid: str, position: str) -> dict[str, float] | None:
    found = vectors["extra_exposure"].get((puuid, position))
    if found is not None:
        return found
    row = vectors["at"].get((puuid, position))
    if row is None or vectors.get("exposure") is None:
        return None
    return dict(zip(vectors["exposure_columns"], vectors["exposure"][row].astype(float)))


def adopt_player(settings: Settings, profile: dict, vectors: dict) -> None:
    from .evaluated import pool_effect, seats_of

    columns = _model_columns(settings)
    pack = _vectors(settings, columns) if columns else None
    if pack is None:
        return
    exposure = vectors.get("exposure")
    with _lock:
        for index, (key, seat) in enumerate(seats_of({**vectors, "puuid": profile["puuid"]}, columns).items()):
            pack["extra"][key] = seat
            champions = (profile.get("champions") or {}).get(key[1], {})
            pack["extra_history"][key] = (0.0, pool_effect(settings, key[1], champions), seat[1])
            if exposure is not None:
                pack["extra_exposure"][key] = dict(zip([str(name) for name in vectors["exposure_columns"]], np.asarray(exposure[index], dtype=float)))


def _effects(path: Path) -> dict[tuple[str, str], float]:
    table = pd.read_parquet(path, columns=["position", "champion", "effect"])
    return {(position, champion): float(effect) for position, champion, effect in zip(table["position"], table["champion"], table["effect"])}


def champion_effects(settings: Settings) -> dict[tuple[str, str], float] | None:
    return cached(settings.served_model_dir / CHAMPION_EFFECTS, "effects", _effects)


def _exposure_of_table(table: pd.DataFrame, path: Path | None) -> tuple[np.ndarray | None, list[str]]:
    if path is None or not path.exists():
        return None, []
    found = pd.read_parquet(path)
    names = [column for column in found.columns if column not in ("puuid", "position")]
    joined = table[["puuid", "position"]].merge(found, on=["puuid", "position"], how="left")
    return joined[names].fillna(0.0).to_numpy(dtype=np.float16), names


def build_vectors(path: Path, columns: list[str], keep=None, exposure_path: Path | None = None) -> dict | None:
    table = pd.read_parquet(path)
    if "position" not in table.columns or any(column not in table.columns for column in columns):
        return None
    matrix = table[columns].to_numpy(dtype=np.float32)
    position = table["position"].to_numpy(dtype=str)
    grid = {name: _grid(matrix[position == name]) for name in np.unique(position)}
    kept = np.ones(len(table), dtype=bool) if keep is None else table["puuid"].isin(keep).to_numpy()
    evidence = table["evidence"].to_numpy(dtype=np.float32) if "evidence" in table.columns else np.full(len(table), np.nan, dtype=np.float32)
    exposure, exposure_columns = _exposure_of_table(table, exposure_path)
    return _pack(
        columns,
        table["puuid"].to_numpy(dtype=str)[kept],
        position[kept],
        table["seats"].to_numpy(dtype=np.int32)[kept],
        evidence[kept],
        matrix[kept],
        grid,
        None if exposure is None else exposure[kept],
        exposure_columns,
    )


def _read_vectors(path: Path) -> dict:
    data = _npz(path)
    grid = {name.removeprefix("grid_"): data[name] for name in data if name.startswith("grid_")}
    exposure = data.get("exposure")
    return _pack(
        [str(name) for name in data["columns"]],
        data["puuid"],
        data["position"],
        data["seats"],
        data["evidence"],
        data["matrix"],
        grid,
        exposure,
        data["exposure_columns"] if exposure is not None else (),
    )


def _profiled(processed_dir: Path) -> set[str] | None:
    path = processed_dir / PROFILES_TABLE
    return set(pd.read_parquet(path, columns=["puuid"])["puuid"]) if path.exists() else None


def _vectors(settings: Settings, columns: list[str]) -> dict | None:
    packed = cached(settings.served_model_dir / VECTORS_FILE, "vectors", _read_vectors)
    if packed is not None and packed["columns"] == list(columns):
        return packed
    kind = f"vectors:{len(columns)}:{hash(tuple(columns))}"
    processed = settings.served_processed_dir
    return cached(processed / STYLES_TABLE, kind, lambda path: build_vectors(path, list(columns), _profiled(processed), processed / EXPOSURE_TABLE))


def write_vectors(model_dir: Path, processed_dir: Path) -> Path | None:
    if not (model_dir / MATRIX_FILE).exists() or not (processed_dir / STYLES_TABLE).exists():
        return None
    saved = _npz(model_dir / MATRIX_FILE)
    if "columns" not in saved:
        return None
    pack = build_vectors(processed_dir / STYLES_TABLE, [str(name) for name in saved["columns"]], _profiled(processed_dir), processed_dir / EXPOSURE_TABLE)
    if pack is None:
        return None
    target = model_dir / VECTORS_FILE
    extra = {} if pack["exposure"] is None else {"exposure": pack["exposure"], "exposure_columns": np.array(pack["exposure_columns"])}
    np.savez(
        target,
        columns=np.array(pack["columns"]),
        puuid=pack["puuid"],
        position=pack["position"],
        seats=pack["seats"],
        evidence=pack["evidence"],
        matrix=pack["matrix"],
        **{f"grid_{name}": grid for name, grid in pack["grid"].items()},
        **extra,
    )
    return target


def _matrix(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / MATRIX_FILE, "npz", _npz)


def _scores(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / SCORES_FILE, "npz", _npz)


def _seats(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / SEATS_FILE, "npz", _npz)


def _informative(settings: Settings) -> bool:
    report = cached(settings.served_model_dir / REPORT_FILE, "json", _json)
    return bool(report and report.get("informative", False))


def _duo_scores(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / DUO_SCORES, "npz", _npz)


def _model_columns(settings: Settings) -> list[str] | None:
    saved = _matrix(settings)
    return [str(name) for name in saved["columns"]] if saved is not None and "columns" in saved else None


def warm(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    _loaded(settings)
    _duo_scores(settings)
    _informative(settings)
    known_names(settings)
    record_between("", "", settings)
    player_history("", "", settings)


def known_names(settings: Settings) -> dict[tuple[str, str], str] | None:
    return cached(settings.served_processed_dir / NAMES_TABLE, "names", _name_index)


class NoGamesInPosition(ValueError):
    def __init__(self, puuid: str, position: str):
        self.puuid, self.position = puuid, position
        super().__init__(f"{puuid} has no games as {spoken(position)} in our data.")


def hinge_between(left: str, right: str, settings: Settings | None = None) -> dict:
    table = cached((settings or get_settings()).served_processed_dir / HINGE_TABLE, "hinge", _indexed_hinge)
    out = {}
    for responder, actor, label in ((left, right, "left_reacts_to_right"), (right, left, "right_reacts_to_left")):
        if table is not None and (responder, actor) in table.index:
            row = table.loc[(responder, actor)]
            out[label] = {name: round(float(row[f"hinge_{name}"]), 3) for name in HINGE_RESPONSES} | {
                "triggers": int(row.hinge_triggers)
            }
        else:
            out[label] = None
    return out


def position_profile(puuid: str, position: str, settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    columns = _model_columns(settings)
    vectors = _vectors(settings, columns) if columns else None
    if vectors is None:
        return None
    seat = seat_of(vectors, puuid, position)
    if seat is None:
        return {"games": 0, "evidence": 0.0}
    return {"games": seat[1], "evidence": _evidence(seat[2])}


def _evidence(value: float) -> float | None:
    return None if np.isnan(value) else round(float(value), 3)


def _quantiles(scores: dict, key: str, combo: str | None) -> np.ndarray:
    combos = [str(name) for name in scores["combos"]] if "combos" in scores else []
    return scores["combo_quantiles"][combos.index(combo)] if combo in combos else scores[key]


def _percentile(value: float, quantiles: np.ndarray) -> float:
    return round(float(np.searchsorted(quantiles, value) / 10.0), 1)


def _score(value: float, quantiles: np.ndarray) -> float:
    spread = float(quantiles.std())
    if spread <= 0.0:
        return 50.0
    return round(float(50.0 + 50.0 * np.tanh((value - float(quantiles.mean())) / (SCORE_SPAN * spread))), 1)


def target_minute(saved: dict) -> int:
    if "minute" in saved:
        return int(saved["minute"])
    units = str(saved.get("units", "gold at 15"))
    return int(units.rsplit(" ", 1)[-1]) if units.rsplit(" ", 1)[-1].isdigit() else 15


def _gold(value: float, scores: dict) -> float:
    return float(value * float(scores["gold_sd"])) if "gold_sd" in scores else float(value)


def _significant(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT}g}")


def _history_index(path: Path) -> dict[tuple[str, str], tuple[float, float, int]]:
    table = pd.read_parquet(path, columns=["puuid", "position", "games", "form", "champion"])
    return {
        (puuid, position): (float(form), float(champion), int(games))
        for puuid, position, games, form, champion in zip(table["puuid"], table["position"], table["games"], table["form"], table["champion"])
    }


def player_history(puuid: str, position: str, settings: Settings) -> tuple[float, float, int] | None:
    found = cached(settings.served_model_dir / PLAYER_HISTORY, "history", _history_index)
    known = (found or {}).get((puuid, position))
    if known is not None:
        return known
    columns = _model_columns(settings)
    pack = _vectors(settings, columns) if columns else None
    return None if pack is None else pack["extra_history"].get((puuid, position))


def _seat_quantiles(settings: Settings, position: str) -> np.ndarray | None:
    scores = _duo_scores(settings)
    if scores is None or "seat_quantiles" not in scores:
        return None
    names = [str(name) for name in scores["seat_positions"]]
    return scores["seat_quantiles"][names.index(position)] if position in names else None


def _seat_calibration(settings: Settings, position: str) -> float:
    scores = _duo_scores(settings)
    if scores is None or "reading_scale" not in scores:
        return 1.0
    names = [str(name) for name in scores["seat_positions"]]
    return float(scores["reading_scale"][names.index(position)]) if position in names else 1.0


def seat_reading(
    seats: dict,
    z: np.ndarray,
    position: str,
    history: tuple[float, float, int] | None = None,
    quantiles: np.ndarray | None = None,
    calibration: float = 1.0,
) -> dict:
    k = [str(name) for name in seats["positions"]].index(position)
    scale = float(seats["scale"][k]) if "scale" in seats else 1.0
    style = calibration * scale * float((z - seats["centres"][k]) @ seats["weights"][k])
    form, champion = (calibration * history[0], calibration * history[1]) if history is not None else (0.0, 0.0)
    gold = style + form + champion
    quantiles = seats["quantiles"][k] if quantiles is None else quantiles
    return {
        "gold": round(gold, 1),
        "score": _score(gold, quantiles),
        "percentile": _percentile(gold, quantiles),
        "style": round(style, 1),
        "form": round(form, 1),
        "champion": round(champion, 1),
    }


def _loaded(settings: Settings) -> tuple[dict, dict, dict, list[str], dict] | None:
    saved, scores, seats = _matrix(settings), _scores(settings), _seats(settings)
    if saved is None or scores is None or seats is None or "columns" not in saved:
        return None
    columns = [str(name) for name in saved["columns"]]
    if [str(name) for name in seats["columns"]] != columns:
        return None
    vectors = _vectors(settings, columns)
    return None if vectors is None else (saved, scores, seats, columns, vectors)


def pair_between(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None
) -> dict | None:
    settings = settings or get_settings()
    if left_position == right_position:
        raise ValueError(f"Both players are set to {spoken(left_position)}. A duo needs two different positions.")
    loaded = _loaded(settings)
    if loaded is None:
        return None
    saved, scores, seats, columns, vectors = loaded
    seat_a, seat_b = seat_of(vectors, left, left_position), seat_of(vectors, right, right_position)
    for puuid, position, seat in ((left, left_position, seat_a), (right, right_position, seat_b)):
        if seat is None:
            raise NoGamesInPosition(puuid, position)
    a, b = seat_a[0], seat_b[0]
    matrix = saved["matrix"]
    dim = matrix.shape[0]
    terms = np.outer(a, b) * matrix / (TEAM_PAIRS * dim)
    value = float(terms.sum())
    strongest = np.argsort(-np.abs(terms).ravel())[:DRIVERS]
    reliable = _informative(settings)
    quantiles = _quantiles(scores, "quantiles", combination(left_position, right_position))
    left_edge = seat_reading(
        seats, a, left_position, player_history(left, left_position, settings), _seat_quantiles(settings, left_position), _seat_calibration(settings, left_position)
    )
    right_edge = seat_reading(
        seats, b, right_position, player_history(right, right_position, settings), _seat_quantiles(settings, right_position), _seat_calibration(settings, right_position)
    )
    gold = _gold(value, scores)
    fit = {"gold": round(gold, 1), "score": _score(value, quantiles), "percentile": _percentile(value, quantiles)}
    return {
        "score": fit["score"],
        "projected_gold": round(gold, 2),
        "minute": target_minute(saved),
        "synergy": value,
        "percentile": fit["percentile"],
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "positions": {"left": left_position, "right": right_position},
        "left_games": seat_a[1],
        "right_games": seat_b[1],
        "left_evidence": _evidence(seat_a[2]),
        "right_evidence": _evidence(seat_b[2]),
        "edge": {
            "left": left_edge,
            "right": right_edge,
            "fit": fit,
            "total": round(left_edge["gold"] + right_edge["gold"] + gold, 1),
        },
        "drivers": [
            {"left": columns[k // dim], "right": columns[k % dim], "contribution": _significant(float(terms.ravel()[k]))}
            for k in strongest
        ],
        "reading": {
            "left": _reading(vectors, left_position, columns, a, terms.sum(axis=1)),
            "right": _reading(vectors, right_position, columns, b, terms.sum(axis=0)),
        },
    }


def _record_index(path: Path) -> dict[tuple[str, str], tuple[int, float, float]]:
    table = pd.read_parquet(path, columns=["a", "b", "games", "mean", "record"])
    return {
        (a, b): (int(games), float(mean), float(record))
        for a, b, games, mean, record in zip(table["a"], table["b"], table["games"], table["mean"], table["record"])
    }


def record_between(left: str, right: str, settings: Settings | None = None, extra: list[float] | tuple[float, ...] = ()) -> dict:
    settings = settings or get_settings()
    records = cached(settings.served_model_dir / DUO_RECORDS, "records", _record_index)
    games, mean, gold = (records or {}).get((left, right) if left < right else (right, left), (0, 0.0, 0.0))
    report = cached(settings.served_model_dir / DUO_REPORT, "json", _json) if extra else None
    counted = 0
    if report and float(report.get("spread", 0.0)) > 0.0 and float(report.get("noise", 0.0)) > 0.0:
        between, noise = float(report["spread"]) ** 2, float(report["noise"])
        counted = len(extra)
        pooled = (mean * games + float(sum(extra))) / (games + counted)
        gold = pooled * between / (between + noise**2 / (games + counted))
    return {"gold": round(gold, 1), "games": games + counted, "customs": counted}


def _expected(loaded, puuid: str, position: str, settings: Settings) -> float:
    _, _, seats, _, vectors = loaded
    seat = seat_of(vectors, puuid, position)
    if seat is None:
        return 0.0
    return seat_reading(seats, seat[0], position, player_history(puuid, position, settings), _seat_quantiles(settings, position), _seat_calibration(settings, position))["gold"]


def custom_residuals(games: list[dict], settings: Settings | None = None) -> list[float]:
    settings = settings or get_settings()
    loaded = _loaded(settings)
    if loaded is None or not games:
        return []
    residuals = []
    for game in games:
        edge = sum(seat["gold"] - seat["enemy_gold"] for seat in game["seats"])
        guess = sum(_expected(loaded, seat["puuid"], seat["position"], settings) - _expected(loaded, seat["enemy"], seat["position"], settings) for seat in game["seats"])
        residuals.append(float(edge - guess))
    return residuals


def duo_between(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None, customs: list[float] | tuple[float, ...] = ()
) -> dict | None:
    settings = settings or get_settings()
    found = pair_between(left, left_position, right, right_position, settings)
    scores = _duo_scores(settings)
    if found is None or scores is None:
        return found
    record = record_between(left, right, settings, customs)
    with_fit = bool(scores["with_fit"]) if "with_fit" in scores else True
    edge = found["edge"]
    total = round(edge["left"]["gold"] + edge["right"]["gold"] + (edge["fit"]["gold"] if with_fit else 0.0) + record["gold"], 1)
    quantiles = _quantiles(scores, "quantiles", combination(left_position, right_position))
    return {
        **found,
        "score": _score(total, quantiles),
        "percentile": _percentile(total, quantiles),
        "projected_gold": total,
        "reliable": found["reliable"] if with_fit else True,
        "note": found["note"] if with_fit else None,
        "edge": {**edge, "record": record, "total": total, "with_fit": with_fit},
    }


def _varies(grid_row: np.ndarray) -> bool:
    if len(grid_row) < MIN_GRID:
        return True
    return bool(grid_row[len(grid_row) * 3 // 4] - grid_row[len(grid_row) // 4] > 1e-9)


def _reading(vectors: dict, position: str, columns: list[str], z: np.ndarray, contributions: np.ndarray) -> dict:
    grid = vectors["grid"][position]
    keep = [index for index, cell in enumerate(columns) if named(cell) and _varies(grid[index])]
    order = sorted(keep, key=lambda index: -abs(z[index]))[:DISTINCTIVE]
    distinctive = [
        {
            "cell": columns[index],
            "words": describe(columns[index]),
            "phrase": phrase(columns[index]),
            "z": round(float(z[index]), 3),
            "percentile": percentile_among(grid[index], z[index]),
        }
        for index in order
    ]
    by_situation: dict[str, float] = {}
    for index in keep:
        situation = situation_of(columns[index])
        by_situation[situation] = by_situation.get(situation, 0.0) + float(contributions[index])
    ranked = sorted(by_situation.items(), key=lambda item: item[1])
    helping = [item for item in reversed(ranked) if item[1] > 0.0][:DISTINCTIVE]
    hurting = [item for item in ranked if item[1] < 0.0][:DISTINCTIVE]
    situations = [
        {"situation": situation, "words": describe_situation(situation), "contribution": _significant(amount)}
        for situation, amount in [*helping, *hurting]
    ]
    return {"distinctive": distinctive, "situations": situations}


def lineup_between(assignments: dict[str, str], settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    if set(assignments) != set(POSITIONS) or len(set(assignments.values())) != len(POSITIONS):
        raise ValueError("a lineup names one distinct player for each of TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY")
    loaded = _loaded(settings)
    if loaded is None:
        return None
    saved, scores, seats, _, vectors = loaded
    pairs = []
    for (left_position, left), (right_position, right) in combinations(
        [(position, assignments[position]) for position in POSITIONS], 2
    ):
        found = pair_between(left, left_position, right, right_position, settings)
        if found is None:
            return None
        pairs.append({"left": left, "right": right, **{k: v for k, v in found.items() if k not in ("drivers", "reading")}})
    total = float(sum(pair["synergy"] for pair in pairs))
    parts = {
        position: seat_reading(
            seats,
            seat_of(vectors, puuid, position)[0],
            position,
            player_history(puuid, position, settings),
            _seat_quantiles(settings, position),
            _seat_calibration(settings, position),
        )["gold"]
        for position, puuid in assignments.items()
    }
    reliable = _informative(settings)
    gold = _gold(total, scores)
    return {
        "score": _score(total, scores["team_quantiles"]),
        "projected_gold": round(gold, 2),
        "minute": target_minute(saved),
        "synergy": total,
        "percentile": _percentile(total, scores["team_quantiles"]),
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "edge": {"seats": parts, "fit": round(gold, 1), "total": round(sum(parts.values()) + gold, 1)},
        "pairs": sorted(pairs, key=lambda pair: -pair["synergy"]),
    }
