import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.describe import describe, describe_situation, named, situation_of
from ..features.hinge import RESPONSES as HINGE_RESPONSES
from ..features.hinge import TABLE as HINGE_TABLE
from ..features.positions import KEY, POSITIONS, combination

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
UNRELIABLE = "the pair term did not beat its shuffled partners on this corpus, so the fit is shown for inspection only"

_held: dict[tuple[str, str], tuple[tuple[int, int], object]] = {}


def cached(path: Path, kind: str, load):
    key = (str(path), kind)
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


def _indexed_styles(path: Path) -> pd.DataFrame | None:
    table = pd.read_parquet(path)
    if "position" not in table.columns:
        return None
    return table.set_index(KEY).sort_index()


def _indexed_hinge(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).set_index(["responder", "actor"]).sort_index()


def _name_index(path: Path) -> dict[tuple[str, str], str]:
    table = pd.read_parquet(path, columns=["puuid", "game_name", "tag_line", "current"])
    table = table.sort_values("current", kind="stable")
    return {
        (str(name).lower(), str(tag).lower()): puuid
        for puuid, name, tag in zip(table["puuid"], table["game_name"], table["tag_line"])
    }


def _styles(settings: Settings) -> pd.DataFrame | None:
    return cached(settings.served_processed_dir / "player_styles.parquet", "styles", _indexed_styles)


def _style_rows(path: Path, columns: list[str]) -> dict:
    table = pd.read_parquet(path, columns=[*KEY, *columns])
    return {
        "at": {key: index for index, key in enumerate(zip(table["puuid"], table["position"]))},
        "matrix": table[columns].to_numpy(dtype=np.float32),
    }


def _vectors(settings: Settings, columns: list[str]) -> dict | None:
    kind = f"vectors:{len(columns)}:{hash(tuple(columns))}"
    return cached(settings.served_processed_dir / "player_styles.parquet", kind, lambda path: _style_rows(path, columns))


def _matrix(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / MATRIX_FILE, "npz", _npz)


def _scores(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / SCORES_FILE, "npz", _npz)


def _seats(settings: Settings) -> dict | None:
    return cached(settings.served_model_dir / SEATS_FILE, "npz", _npz)


def _informative(settings: Settings) -> bool:
    report = cached(settings.served_model_dir / REPORT_FILE, "json", _json)
    return bool(report and report.get("informative", False))


def known_names(settings: Settings) -> dict[tuple[str, str], str] | None:
    return cached(settings.served_processed_dir / NAMES_TABLE, "names", _name_index)


class NoGamesInPosition(ValueError):
    def __init__(self, puuid: str, position: str):
        self.puuid, self.position = puuid, position
        super().__init__(f"{puuid} has no games as {position} in the corpus")


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
    styles = _styles(settings or get_settings())
    if styles is None:
        return None
    if (puuid, position) not in styles.index:
        return {"games": 0, "evidence": 0.0}
    row = styles.loc[(puuid, position)]
    evidence = float(row["evidence"]) if "evidence" in styles.columns and pd.notna(row["evidence"]) else None
    return {"games": int(row["seats"]), "evidence": None if evidence is None else round(evidence, 3)}


def _evidence(styles: pd.DataFrame, puuid: str, position: str) -> float | None:
    if "evidence" not in styles.columns or pd.isna(styles.loc[(puuid, position), "evidence"]):
        return None
    return round(float(styles.loc[(puuid, position), "evidence"]), 3)


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


def _gold(value: float, scores: dict) -> float:
    return float(value * float(scores["gold_sd"])) if "gold_sd" in scores else float(value)


def _significant(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT}g}")


def seat_reading(seats: dict, z: np.ndarray, position: str) -> dict:
    k = [str(name) for name in seats["positions"]].index(position)
    scale = float(seats["scale"][k]) if "scale" in seats else 1.0
    gold = scale * float((z - seats["centres"][k]) @ seats["weights"][k])
    quantiles = seats["quantiles"][k]
    return {"gold": round(gold, 1), "score": _score(gold, quantiles), "percentile": _percentile(gold, quantiles)}


def _loaded(settings: Settings) -> tuple[pd.DataFrame, dict, dict, dict, list[str], dict] | None:
    styles, saved, scores, seats = _styles(settings), _matrix(settings), _scores(settings), _seats(settings)
    if styles is None or saved is None or scores is None or seats is None or "columns" not in saved:
        return None
    columns = [str(name) for name in saved["columns"]]
    if any(name not in styles.columns for name in columns) or [str(name) for name in seats["columns"]] != columns:
        return None
    return styles, saved, scores, seats, columns, _vectors(settings, columns)


def pair_between(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None
) -> dict | None:
    settings = settings or get_settings()
    if left_position == right_position:
        raise ValueError(f"both players are given {left_position}, a duo needs two different positions")
    loaded = _loaded(settings)
    if loaded is None:
        return None
    styles, saved, scores, seats, columns, vectors = loaded
    for puuid, position in ((left, left_position), (right, right_position)):
        if (puuid, position) not in vectors["at"]:
            raise NoGamesInPosition(puuid, position)
    a = vectors["matrix"][vectors["at"][(left, left_position)]].astype(float)
    b = vectors["matrix"][vectors["at"][(right, right_position)]].astype(float)
    matrix = saved["matrix"]
    dim = matrix.shape[0]
    terms = np.outer(a, b) * matrix / (TEAM_PAIRS * dim)
    value = float(terms.sum())
    strongest = np.argsort(-np.abs(terms).ravel())[:DRIVERS]
    reliable = _informative(settings)
    quantiles = _quantiles(scores, "quantiles", combination(left_position, right_position))
    left_edge, right_edge = seat_reading(seats, a, left_position), seat_reading(seats, b, right_position)
    gold = _gold(value, scores)
    fit = {"gold": round(gold, 1), "score": _score(value, quantiles), "percentile": _percentile(value, quantiles)}
    return {
        "score": fit["score"],
        "projected_gold_at_15": round(gold, 2),
        "synergy": value,
        "percentile": fit["percentile"],
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "positions": {"left": left_position, "right": right_position},
        "left_games": int(styles.loc[(left, left_position), "seats"]),
        "right_games": int(styles.loc[(right, right_position), "seats"]),
        "left_evidence": _evidence(styles, left, left_position),
        "right_evidence": _evidence(styles, right, right_position),
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
            "left": _reading(styles, left_position, columns, a, terms.sum(axis=1)),
            "right": _reading(styles, right_position, columns, b, terms.sum(axis=0)),
        },
    }


def _reading(styles: pd.DataFrame, position: str, columns: list[str], z: np.ndarray, contributions: np.ndarray) -> dict:
    peers = styles.index.get_level_values("position") == position
    keep = [index for index, cell in enumerate(columns) if named(cell)]
    order = sorted(keep, key=lambda index: -abs(z[index]))[:DISTINCTIVE]
    distinctive = [
        {
            "cell": columns[index],
            "words": describe(columns[index]),
            "z": round(float(z[index]), 3),
            "percentile": round(float((styles[columns[index]].to_numpy(dtype=float)[peers] < z[index]).mean() * 100.0), 1),
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
    _, _, scores, seats, _, vectors = loaded
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
        position: seat_reading(seats, vectors["matrix"][vectors["at"][(puuid, position)]].astype(float), position)["gold"]
        for position, puuid in assignments.items()
    }
    reliable = _informative(settings)
    gold = _gold(total, scores)
    return {
        "score": _score(total, scores["team_quantiles"]),
        "projected_gold_at_15": round(gold, 2),
        "synergy": total,
        "percentile": _percentile(total, scores["team_quantiles"]),
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "edge": {"seats": parts, "fit": round(gold, 1), "total": round(sum(parts.values()) + gold, 1)},
        "pairs": sorted(pairs, key=lambda pair: -pair["synergy"]),
    }
