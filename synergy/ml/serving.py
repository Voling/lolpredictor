import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from ..config import Settings, get_settings
from ..features.describe import describe, named
from ..features.hinge import RESPONSES as HINGE_RESPONSES
from ..features.hinge import TABLE as HINGE_TABLE
from ..features.positions import KEY, POSITIONS

DRIVERS = 6
DRIVER_REFERENCE = 32
DISTINCTIVE = 5
SCORE_SPAN = 6.0
SIGNIFICANT = 4
NAMES_TABLE = "player_names.parquet"
NETWORK_FILE = "pairnet.npz"
SEATS_FILE = "seat_weights.npz"
REPORT_FILE = "pairnet_report.json"
UNRELIABLE = "the pair network did not beat its shuffled partners on this corpus, so the fit is shown for inspection only"
FAMILY_WORDS = {
    "tend": "fight and lane tendencies",
    "prio": "lane state",
    "rsp": "reactions after kills, plates and buildings",
    "obj": "play at dragons and void grubs",
    "ward": "warding",
    "jgl": "jungle openings",
    "habit": "habit components",
    "move": "movement components",
    "style": "embedding components",
}

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


def _style_rows(path: Path, columns: list[str]) -> dict:
    table = pd.read_parquet(path, columns=[*KEY, *columns])
    return {
        "at": {key: index for index, key in enumerate(zip(table["puuid"], table["position"]))},
        "matrix": table[columns].to_numpy(dtype=np.float32),
    }


def _vectors(settings: Settings, columns: list[str]) -> dict | None:
    kind = f"vectors:{len(columns)}:{hash(tuple(columns))}"
    return cached(settings.processed_dir / "player_styles.parquet", kind, lambda path: _style_rows(path, columns))


def _network(settings: Settings) -> dict | None:
    return cached(settings.model_dir / NETWORK_FILE, "npz", _npz)


def _seats(settings: Settings) -> dict | None:
    return cached(settings.model_dir / SEATS_FILE, "npz", _npz)


def _informative(settings: Settings) -> bool:
    report = cached(settings.model_dir / REPORT_FILE, "json", _json)
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


def _percentile(value: float, quantiles: np.ndarray) -> float:
    return round(float(np.searchsorted(quantiles, value) / 10.0), 1)


def _score(value: float, quantiles: np.ndarray) -> float:
    spread = float(quantiles.std())
    if spread <= 0.0:
        return 50.0
    return round(float(50.0 + 50.0 * np.tanh((value - float(quantiles.mean())) / (SCORE_SPAN * spread))), 1)


def _significant(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT}g}")


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + erf(values / np.sqrt(2.0)))


def network_sides(net: dict, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    rows = np.concatenate([first, second, first * second], axis=1).astype(np.float32)
    out = np.zeros((len(net["scale"]), len(rows)))
    for seed in range(len(net["scale"])):
        hidden = _gelu(rows @ net["first_weight"][seed].T + net["first_bias"][seed])
        hidden = _gelu(hidden @ net["second_weight"][seed].T + net["second_bias"][seed])
        out[seed] = (hidden @ net["third_weight"][seed] + net["third_bias"][seed]) * float(net["scale"][seed])
    return out


def network_side(net: dict, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return network_sides(net, first, second).mean(axis=0)


def _combo(net: dict, first_index: int, second_index: int) -> int:
    return [str(name) for name in net["combos"]].index(f"{POSITIONS[first_index]}+{POSITIONS[second_index]}")


def _band(net: dict, known: float) -> int:
    return int(np.searchsorted(net["evidence_bands"], known, side="right") - 1)


def fits_between(net: dict, first: np.ndarray, second: np.ndarray, first_index: int, second_index: int) -> tuple[dict[str, float], float]:
    columns = [str(name) for name in net["columns"]]
    families = [str(name) for name in net["families"]]
    combo = _combo(net, first_index, second_index)
    reference_first, reference_second = net["reference"][first_index], net["reference"][second_index]
    blocks = [("all", None, len(reference_first), float(net["grand"][combo]))]
    for family, grand in zip(families, net["grand_without"][combo]):
        mask = np.array([0.0 if name.split("_")[0] == family else 1.0 for name in columns], dtype=np.float32)
        blocks.append((family, mask, min(DRIVER_REFERENCE, len(reference_first)), float(grand)))
    firsts, seconds = [], []
    for _, mask, count, _ in blocks:
        a, b = (first, second) if mask is None else (first * mask, second * mask)
        others_a, others_b = reference_first[:count], reference_second[:count]
        if mask is not None:
            others_a, others_b = others_a * mask, others_b * mask
        firsts += [a[None], np.repeat(a[None], count, axis=0), others_a]
        seconds += [b[None], others_b, np.repeat(b[None], count, axis=0)]
    seeds = network_sides(net, np.concatenate(firsts), np.concatenate(seconds))
    values = seeds.mean(axis=0)
    out, start, error = {}, 0, 0.0
    for name, _, count, grand in blocks:
        block = values[start : start + 1 + 2 * count]
        out[name] = float(block[0] - block[1 : 1 + count].mean() - block[1 + count :].mean() + grand)
        if name == "all" and len(seeds) > 1:
            each = seeds[:, start : start + 1 + 2 * count]
            by_seed = each[:, 0] - each[:, 1 : 1 + count].mean(axis=1) - each[:, 1 + count :].mean(axis=1) + net["grand_seeds"][combo]
            error = float(by_seed.std(ddof=1))
        start += 1 + 2 * count
    return out, error


def seat_reading(seats: dict, z: np.ndarray, position: str) -> dict:
    k = [str(name) for name in seats["positions"]].index(position)
    gold = float((z - seats["centres"][k]) @ seats["weights"][k])
    quantiles = seats["quantiles"][k]
    return {"gold": round(gold, 1), "score": _score(gold, quantiles), "percentile": _percentile(gold, quantiles)}


def _loaded(settings: Settings) -> tuple[pd.DataFrame, dict, dict, list[str], dict] | None:
    styles, net, seats = _styles(settings), _network(settings), _seats(settings)
    if styles is None or net is None or seats is None:
        return None
    columns = [str(name) for name in net["columns"]]
    if any(name not in styles.columns for name in columns) or [str(name) for name in seats["columns"]] != columns:
        return None
    return styles, net, seats, columns, _vectors(settings, columns)


def pair_between(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None
) -> dict | None:
    settings = settings or get_settings()
    if left_position == right_position:
        raise ValueError(f"both players are given {left_position}, a duo needs two different positions")
    loaded = _loaded(settings)
    if loaded is None:
        return None
    styles, net, seats, columns, vectors = loaded
    for puuid, position in ((left, left_position), (right, right_position)):
        if (puuid, position) not in vectors["at"]:
            raise NoGamesInPosition(puuid, position)
    a = vectors["matrix"][vectors["at"][(left, left_position)]]
    b = vectors["matrix"][vectors["at"][(right, right_position)]]
    i, j = POSITIONS.index(left_position), POSITIONS.index(right_position)
    fits, error = fits_between(net, a, b, i, j) if i < j else fits_between(net, b, a, j, i)
    value = fits["all"]
    left_games = int(styles.loc[(left, left_position), "seats"])
    right_games = int(styles.loc[(right, right_position), "seats"])
    left_evidence, right_evidence = _evidence(styles, left, left_position), _evidence(styles, right, right_position)
    band = _band(net, (left_evidence or 0.0) * (right_evidence or 0.0))
    quantiles = net["fit_quantiles"][_combo(net, min(i, j), max(i, j))][band]
    reliable = _informative(settings)
    left_edge, right_edge = seat_reading(seats, a, left_position), seat_reading(seats, b, right_position)
    fit = {
        "gold": round(value, 1),
        "score": _score(value, quantiles),
        "percentile": _percentile(value, quantiles),
        "error": round(error, 1),
        "known": float(net["evidence_bands"][band]),
    }
    drivers = sorted(((family, value - fits[family]) for family in fits if family != "all"), key=lambda item: -abs(item[1]))
    return {
        "score": fit["score"],
        "projected_gold_at_15": round(value, 2),
        "synergy": value,
        "percentile": fit["percentile"],
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "positions": {"left": left_position, "right": right_position},
        "left_games": left_games,
        "right_games": right_games,
        "left_evidence": left_evidence,
        "right_evidence": right_evidence,
        "edge": {
            "left": left_edge,
            "right": right_edge,
            "fit": fit,
            "total": round(left_edge["gold"] + right_edge["gold"] + value, 1),
        },
        "drivers": [
            {"family": family, "words": FAMILY_WORDS.get(family, family), "contribution": _significant(amount)}
            for family, amount in drivers[:DRIVERS]
        ],
        "reading": {
            "left": _reading(styles, left_position, columns, a),
            "right": _reading(styles, right_position, columns, b),
        },
    }


def _reading(styles: pd.DataFrame, position: str, columns: list[str], z: np.ndarray) -> dict:
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
    return {"distinctive": distinctive}


def lineup_between(assignments: dict[str, str], settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    if set(assignments) != set(POSITIONS) or len(set(assignments.values())) != len(POSITIONS):
        raise ValueError("a lineup names one distinct player for each of TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY")
    loaded = _loaded(settings)
    if loaded is None:
        return None
    _, net, seats, _, vectors = loaded
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
        position: seat_reading(seats, vectors["matrix"][vectors["at"][(puuid, position)]], position)["gold"]
        for position, puuid in assignments.items()
    }
    reliable = _informative(settings)
    quantiles = net["team_quantiles"]
    return {
        "score": _score(total, quantiles),
        "projected_gold_at_15": round(total, 2),
        "synergy": total,
        "percentile": _percentile(total, quantiles),
        "reliable": reliable,
        "note": None if reliable else UNRELIABLE,
        "edge": {"seats": parts, "fit": round(total, 1), "total": round(sum(parts.values()) + total, 1)},
        "pairs": sorted(pairs, key=lambda pair: -pair["synergy"]),
    }
