from pathlib import Path

import numpy as np
from scipy import stats

from ..config import Settings, get_settings
from ..features.describe import describe_situation, outcome_words, tendency_words
from ..features.fits import BUNDLE, PRIORITY_FIT, PRIORS_FILE, REACTION_FITS, load_cells
from ..features.priority import BANDS, FARMING, OPPONENT
from ..features.propensity import KINDS
from ..features.tendency import CELLS
from .serving import _json, _model_columns, _npz, _vectors, cached, exposure_of, seat_of

DRAWS = 400
TOP = 3
RANGE = (10.0, 90.0)
FLOOR = 1e-3
RARE = 0.01
MEASURED_GAMES = 20
MEASURED_PLAYERS = 100
STANDARDISE = "standardise.npz"
LANE_GROUPS = {**{f"{own}_{opponent}_{farm}": own for own in BANDS for opponent in OPPONENT for farm in FARMING}, "off_lane": "off_lane", "dead": "dead"}
LANE_ORDER = [*BANDS, "off_lane", "dead"]


def _bundle_dir(settings: Settings) -> Path:
    return settings.served_model_dir.parent / BUNDLE


def fits_for(settings: Settings) -> dict | None:
    folder = _bundle_dir(settings)
    cells = [cached(folder / name, "cells", load_cells) for name in (PRIORITY_FIT, *REACTION_FITS)]
    standard = cached(folder / STANDARDISE, "npz", _npz)
    if any(fit is None for fit in cells) or standard is None:
        return None
    return {"cells": cells, "standard": standard, "priors": cached(folder / PRIORS_FILE, "json", _json) or {}}


def _cell_index(fits: list[dict]) -> dict[str, tuple[dict, int, int]]:
    return {
        f"{fit['prefix']}_{situation}_{outcome}": (fit, step, k)
        for fit in fits
        for step, situation in enumerate(fit["situations"])
        for k, outcome in enumerate(fit["outcomes"])
    }


def _lane_situation(cell: str) -> str:
    return "prio_pre_objective" if cell.startswith("prio_pre_objective_") else "prio_all"


def measured_lanes(pack: dict, position: str, columns: list[str]) -> dict[int, np.ndarray]:
    held = pack.setdefault("measured", {})
    if position in held:
        return held[position]
    found = {}
    names = pack.get("exposure_columns") or []
    if pack.get("exposure") is not None:
        rows = np.flatnonzero(np.asarray(pack["position"]) == position)
        for column, cell in enumerate(columns):
            situation = _lane_situation(cell)
            if not cell.startswith("prio_") or situation not in names:
                continue
            many = rows[np.asarray(pack["exposure"][rows, names.index(situation)], dtype=float) >= MEASURED_GAMES]
            if len(many) >= MEASURED_PLAYERS:
                found[column] = np.sort(np.asarray(pack["matrix"][many, column], dtype=float))
    held[position] = found
    return found


def population_percentiles(z, columns: list[str], position: str, fits: dict, pack: dict | None = None, exposure: dict | None = None) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    standard = fits["standard"]
    at = {str(name): index for index, name in enumerate(standard["columns"])}
    index = _cell_index(fits["cells"])
    lanes = {} if pack is None else measured_lanes(pack, position, columns)
    out = np.full(len(columns), np.nan)
    shares, ratios = [], []
    for column, cell in enumerate(columns):
        if cell not in at:
            continue
        value = z[column] * float(standard["spread"][at[cell]]) + float(standard["centre"][at[cell]])
        if cell in index:
            fit, step, k = index[cell]
            if position not in fit["positions"]:
                continue
            share = float(fit["worlds"][step, fit["positions"].index(position), k])
            if share < RARE:
                continue
            if cell.startswith("prio_"):
                measured = float((exposure or {}).get(_lane_situation(cell), 0.0) or 0.0) >= MEASURED_GAMES
                if column in lanes and measured:
                    out[column] = 100.0 * np.searchsorted(lanes[column], z[column], side="left") / len(lanes[column])
            else:
                shares.append((column, min(max(value, 0.0), 1.0), float(fit["kappa"][step]), share))
        elif cell.startswith("tend_"):
            prior = fits["priors"].get("_".join(cell.split("_")[1:-2]))
            if prior:
                ratios.append((column, float(np.exp(value)), float(prior)))
    if shares:
        columns_, values, kappas, priors = map(np.array, zip(*shares))
        out[columns_.astype(int)] = 100.0 * stats.beta.cdf(values, kappas * priors, kappas * (1.0 - priors))
    if ratios:
        columns_, values, priors = map(np.array, zip(*ratios))
        out[columns_.astype(int)] = 100.0 * stats.gamma.cdf(values, priors, scale=1.0 / priors)
    return out


def _collapse(alpha: np.ndarray, prior: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = [LANE_GROUPS.get(name, name) for name in names]
    order = [group for group in LANE_ORDER if group in groups]
    picks = [[index for index, group in enumerate(groups) if group == wanted] for wanted in order]
    return np.array([alpha[pick].sum() for pick in picks]), np.array([prior[pick].sum() for pick in picks]), order


def situations_from(z, columns: list[str], exposure: dict, position: str, fits: list[dict], priors: dict, standard: dict) -> list[dict]:
    z = np.asarray(z, dtype=float)
    at = {column: index for index, column in enumerate(columns)}
    standard_at = {str(name): index for index, name in enumerate(standard["columns"])}
    centre, spread = np.asarray(standard["centre"], dtype=float), np.asarray(standard["spread"], dtype=float)
    found = []
    for fit in fits:
        if position not in fit["positions"]:
            continue
        code = fit["positions"].index(position)
        prefix, outcomes = fit["prefix"], list(fit["outcomes"])
        for step, situation in enumerate(fit["situations"]):
            key = f"{prefix}_{situation}"
            chances = float(exposure.get(key, 0.0) or 0.0)
            cells = [f"{key}_{outcome}" for outcome in outcomes]
            if chances <= 0.0 or any(cell not in at or cell not in standard_at for cell in cells):
                continue
            rows = [at[cell] for cell in cells]
            standard_rows = [standard_at[cell] for cell in cells]
            mean = np.clip(z[rows] * spread[standard_rows] + centre[standard_rows], FLOOR, None)
            alpha = mean / mean.sum() * (chances + float(fit["kappa"][step]))
            prior = np.asarray(fit["worlds"][step, code], dtype=float)
            names = outcomes
            if prefix == "prio":
                alpha, prior, names = _collapse(alpha, prior, names)
            found.append({"situation": key, "words": describe_situation(key), "kind": "shares", "n": round(chances, 1), "alpha": alpha, "prior": prior, "outcomes": names})
    for kind in KINDS:
        prior = priors.get(kind)
        if prior is None:
            continue
        for cell in CELLS:
            key = f"tend_{kind}_{cell}"
            observed = float(exposure.get(f"{key}_obs", 0.0) or 0.0)
            expected = float(exposure.get(f"{key}_exp", 0.0) or 0.0)
            if expected <= 0.0 or key not in at:
                continue
            pattern, half = cell.rsplit("_", 1)
            found.append({"situation": key, "words": tendency_words(kind, pattern, half), "kind": "ratio", "n": round(expected, 1), "observed": observed, "expected": expected, "prior": float(prior)})
    return found


def _band(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(draws, RANGE, axis=0)
    return low, high


def evidence(entry: dict, rng: np.random.Generator) -> float:
    if entry["kind"] == "ratio":
        draws = np.log(rng.gamma(entry["observed"] + entry["prior"], 1.0 / (entry["expected"] + entry["prior"]), DRAWS))
        spread = float(draws.std())
        return abs(float(draws.mean())) / spread if spread > 0.0 else 0.0
    draws = rng.dirichlet(np.maximum(entry["alpha"], FLOOR), DRAWS)
    spread = draws.std(axis=0)
    gap = np.abs(draws.mean(axis=0) - entry["prior"])
    return float(np.max(np.where(spread > 0.0, gap / np.where(spread > 0.0, spread, 1.0), 0.0)))


def _shown(name: str, high: float, prior: float, *others: float) -> bool:
    return max(high, prior, *others) >= FLOOR * 10 or name in ("dead", "off_lane")


def summarise(entry: dict, rng: np.random.Generator) -> dict:
    if entry["kind"] == "ratio":
        shape, rate = entry["observed"] + entry["prior"], entry["expected"] + entry["prior"]
        low, high = _band(rng.gamma(shape, 1.0 / rate, DRAWS))
        return {
            "situation": entry["situation"],
            "words": entry["words"],
            "kind": "ratio",
            "n": entry["n"],
            "observed": round(entry["observed"], 1),
            "expected": round(entry["expected"], 1),
            "ratio": {"mean": round(shape / rate, 3), "low": round(float(low), 3), "high": round(float(high), 3)},
        }
    alpha = np.maximum(entry["alpha"], FLOOR)
    low, high = _band(rng.dirichlet(alpha, DRAWS))
    mean = alpha / alpha.sum()
    return {
        "situation": entry["situation"],
        "words": entry["words"],
        "kind": "shares",
        "n": entry["n"],
        "outcomes": [
            {"name": name, "words": outcome_words(name), "mean": round(float(mean[k]), 4), "low": round(float(low[k]), 4), "high": round(float(high[k]), 4), "prior": round(float(entry["prior"][k]), 4)}
            for k, name in enumerate(entry["outcomes"])
            if _shown(name, float(high[k]), float(entry["prior"][k]))
        ],
    }


def top_situations(entries: list[dict], rng: np.random.Generator, top: int = TOP) -> list[dict]:
    ranked = sorted(entries, key=lambda entry: -evidence(entry, rng))[:top]
    return [summarise(entry, rng) for entry in ranked]


def differences(left: list[dict], right: list[dict], rng: np.random.Generator, top: int = TOP) -> list[dict]:
    right_by = {entry["situation"]: entry for entry in right if entry["kind"] == "shares"}
    scored = []
    for a in left:
        b = right_by.get(a["situation"])
        if a["kind"] != "shares" or b is None or a["outcomes"] != b["outcomes"]:
            continue
        draws_a = rng.dirichlet(np.maximum(a["alpha"], FLOOR), DRAWS)
        draws_b = rng.dirichlet(np.maximum(b["alpha"], FLOOR), DRAWS)
        scored.append((float(np.minimum(draws_a, draws_b).sum(axis=1).mean()), a, b, draws_a, draws_b))
    scored.sort(key=lambda item: item[0])
    out = []
    for overlap, a, b, draws_a, draws_b in scored[:top]:
        low_a, high_a = _band(draws_a)
        low_b, high_b = _band(draws_b)
        mean_a, mean_b = a["alpha"] / a["alpha"].sum(), b["alpha"] / b["alpha"].sum()
        out.append(
            {
                "situation": a["situation"],
                "words": a["words"],
                "overlap": round(overlap, 3),
                "outcomes": [
                    {
                        "name": name,
                        "words": outcome_words(name),
                        "left": {"mean": round(float(mean_a[k]), 4), "low": round(float(low_a[k]), 4), "high": round(float(high_a[k]), 4)},
                        "right": {"mean": round(float(mean_b[k]), 4), "low": round(float(low_b[k]), 4), "high": round(float(high_b[k]), 4)},
                    }
                    for k, name in enumerate(a["outcomes"])
                    if _shown(name, float(high_a[k]), float(a["prior"][k]), float(high_b[k]))
                ],
            }
        )
    return out


def player_entries(puuid: str, position: str, settings: Settings) -> list[dict]:
    columns = _model_columns(settings)
    pack = _vectors(settings, columns) if columns else None
    fits = fits_for(settings)
    if pack is None or fits is None:
        return []
    seat, exposure = seat_of(pack, puuid, position), exposure_of(pack, puuid, position)
    if seat is None or exposure is None:
        return []
    return situations_from(seat[0], pack["columns"], exposure, position, fits["cells"], fits["priors"], fits["standard"])


def duo_posteriors(left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    rng = np.random.default_rng(7)
    a, b = player_entries(left, left_position, settings), player_entries(right, right_position, settings)
    return {"left": top_situations(a, rng), "right": top_situations(b, rng), "differences": differences(a, b, rng)}
