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
UNRELIABLE = "the pair term did not beat its nulls on this corpus, so the percentile is shown for inspection only"

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
    return cached(settings.processed_dir / "player_styles.parquet", "styles", _indexed_styles)


def _matrix(settings: Settings) -> dict | None:
    return cached(settings.model_dir / "interaction_matrix.npz", "npz", _npz)


def _scores(settings: Settings) -> dict | None:
    return cached(settings.model_dir / "interaction_scores.npz", "npz", _npz)


def _informative(settings: Settings) -> bool:
    report = cached(settings.model_dir / "interaction_report.json", "json", _json)
    return bool(report and report.get("informative", False))


def known_names(settings: Settings) -> dict[tuple[str, str], str] | None:
    return cached(settings.processed_dir / NAMES_TABLE, "names", _name_index)


class NoGamesInPosition(ValueError):
    def __init__(self, puuid: str, position: str):
        self.puuid, self.position = puuid, position
        super().__init__(f"{puuid} has no games as {position} in the corpus")


def hinge_between(left: str, right: str, settings: Settings | None = None) -> dict:
    table = cached((settings or get_settings()).processed_dir / HINGE_TABLE, "hinge", _indexed_hinge)
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


def _gold(value: float, scores: dict) -> float | None:
    return round(value * float(scores["gold_sd"]), 2) if "gold_sd" in scores else None


def _significant(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT}g}")


def pair_between(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None
) -> dict | None:
    settings = settings or get_settings()
    if left_position == right_position:
        raise ValueError(f"both players are given {left_position}, a duo needs two different positions")
    styles, saved, scores = _styles(settings), _matrix(settings), _scores(settings)
    if styles is None or saved is None or scores is None or "columns" not in saved:
        return None
    matrix, columns = saved["matrix"], [str(c) for c in saved["columns"]]
    if any(c not in styles.columns for c in columns):
        return None
    for puuid, position in ((left, left_position), (right, right_position)):
        if (puuid, position) not in styles.index:
            raise NoGamesInPosition(puuid, position)
    a = styles.loc[(left, left_position), columns].to_numpy(dtype=float)
    b = styles.loc[(right, right_position), columns].to_numpy(dtype=float)
    dim = matrix.shape[0]
    terms = np.outer(a, b) * matrix / (TEAM_PAIRS * dim)
    value = float(terms.sum())
    strongest = np.argsort(-np.abs(terms).ravel())[:DRIVERS]
    reliable = _informative(settings)
    quantiles = _quantiles(scores, "quantiles", combination(left_position, right_position))
    return {
        "score": _score(value, quantiles),
        "projected_gold_at_15": _gold(value, scores),
        "synergy": value,
        "percentile": _percentile(value, quantiles),
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "positions": {"left": left_position, "right": right_position},
        "left_games": int(styles.loc[(left, left_position), "seats"]),
        "right_games": int(styles.loc[(right, right_position), "seats"]),
        "left_evidence": _evidence(styles, left, left_position),
        "right_evidence": _evidence(styles, right, right_position),
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
    pairs = []
    for (left_position, left), (right_position, right) in combinations(
        [(position, assignments[position]) for position in POSITIONS], 2
    ):
        found = pair_between(left, left_position, right, right_position, settings)
        if found is None:
            return None
        pairs.append({"left": left, "right": right, **{k: v for k, v in found.items() if k not in ("drivers", "reading")}})
    total = float(sum(pair["synergy"] for pair in pairs))
    reliable = _informative(settings)
    scores = _scores(settings)
    return {
        "score": _score(total, scores["team_quantiles"]),
        "projected_gold_at_15": _gold(total, scores),
        "synergy": total,
        "percentile": _percentile(total, scores["team_quantiles"]),
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "pairs": sorted(pairs, key=lambda pair: -pair["synergy"]),
    }
