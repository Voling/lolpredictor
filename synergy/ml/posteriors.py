from pathlib import Path

import numpy as np
from scipy import stats

from ..config import Settings, get_settings
from ..features.describe import act, describe_situation, outcome_label, phrase, situation_lead, tendency_words
from ..features.exposure import exposure_column
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
CLUSTERED = 5.0
SURE = 0.9
GAP = 0.03
ROLES = {"TOP": "top laner", "JUNGLE": "jungler", "MIDDLE": "mid laner", "BOTTOM": "bot laner", "UTILITY": "support"}
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


def _readings(fit: dict) -> list[tuple[str, int, list[str] | None]]:
    found = [(f"{fit['prefix']}_{situation}", step, None) for step, situation in enumerate(fit["situations"])]
    split = fit.get("split")
    if split:
        found += [(f"{split['prefix']}_{group}_{situation}", fit["situations"].index(situation), states) for group, states in split["groups"].items() for situation in split["situations"]]
    return found


def _cell_index(fits: list[dict]) -> dict[str, tuple[dict, int, int, list[str] | None]]:
    return {f"{key}_{outcome}": (fit, step, k, states) for fit in fits for key, step, states in _readings(fit) for k, outcome in enumerate(fit["outcomes"])}


def trusted(fit: dict, step: int) -> bool:
    return fit["prefix"] == "prio" or float(fit["kappa"][step]) >= CLUSTERED


def _lane_situation(cell: str) -> str:
    return "prio_pre_objective" if cell.startswith("prio_pre_objective_") else "prio_all"


def expected_mix(fit: dict, step: int, code: int, exposure: dict, chosen: list[str] | None = None) -> tuple[float, np.ndarray | None]:
    key, states = f"{fit['prefix']}_{fit['situations'][step]}", list(fit["states"])
    chosen = states if chosen is None else chosen
    worlds = np.asarray(fit["worlds"][step, code], dtype=float)[[states.index(state) for state in chosen]]
    chances = np.array([float(exposure.get(exposure_column(key, state, states), 0.0) or 0.0) for state in chosen])
    total = float(chances.sum())
    if len(states) == 1:
        return total, worlds[0]
    return total, (chances @ worlds / total if total > 0.0 else None)


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
            fit, step, k, chosen = index[cell]
            if position not in fit["positions"]:
                continue
            mix = expected_mix(fit, step, fit["positions"].index(position), exposure or {}, chosen)[1]
            if mix is None or mix[k] < RARE or trusted(fit, step) is False:
                continue
            if cell.startswith("prio_"):
                measured = float((exposure or {}).get(_lane_situation(cell), 0.0) or 0.0) >= MEASURED_GAMES
                if column in lanes and measured:
                    out[column] = 100.0 * np.searchsorted(lanes[column], z[column], side="left") / len(lanes[column])
            else:
                share = value + mix[k] if fit["gap"] else value
                shares.append((column, min(max(share, 0.0), 1.0), float(fit["kappa"][step]), float(mix[k])))
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
        for key, step, chosen in _readings(fit):
            if trusted(fit, step) is False:
                continue
            chances, prior = expected_mix(fit, step, code, exposure, chosen)
            cells = [f"{key}_{outcome}" for outcome in outcomes]
            if chances <= 0.0 or any(cell not in at or cell not in standard_at for cell in cells):
                continue
            rows = [at[cell] for cell in cells]
            standard_rows = [standard_at[cell] for cell in cells]
            served = z[rows] * spread[standard_rows] + centre[standard_rows]
            mean = np.clip(served + prior if fit["gap"] else served, FLOOR, None)
            alpha = mean / mean.sum() * (chances + float(fit["kappa"][step]))
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


def _pct(value: float) -> str:
    return f"{round(float(value) * 100)}%"


def short_name(riot_id: str | None) -> str:
    return str(riot_id).split("#")[0] if riot_id else "This player"


def _size(gap: float, mean: float, prior: float) -> str:
    if abs(gap) >= 0.15 or (prior >= 0.02 and (mean >= 2 * prior or mean <= prior / 2)):
        return "far "
    return "slightly " if abs(gap) < 0.05 else ""


def clear_gap(mean: np.ndarray, prior: np.ndarray, draws: np.ndarray) -> tuple[int | None, float]:
    best, best_gap = None, 0.0
    for k in range(len(mean)):
        gap = float(mean[k] - prior[k])
        sure = float((draws[:, k] > prior[k]).mean() if gap > 0 else (draws[:, k] < prior[k]).mean())
        if sure >= SURE and abs(gap) >= GAP and abs(gap) > abs(best_gap):
            best, best_gap = k, gap
    return best, best_gap


def _draws(entry: dict, rng: np.random.Generator) -> np.ndarray:
    if entry["kind"] == "ratio":
        return rng.gamma(entry["observed"] + entry["prior"], 1.0 / (entry["expected"] + entry["prior"]), DRAWS)
    return rng.dirichlet(np.maximum(entry["alpha"], FLOOR), DRAWS)


def _clearness(entry: dict, draws: np.ndarray) -> tuple[float, float]:
    if entry["kind"] == "ratio":
        shape, rate = entry["observed"] + entry["prior"], entry["expected"] + entry["prior"]
        low, high = _band(draws)
        return 0.0, float(abs(np.log(shape / rate))) if low > 1.0 or high < 1.0 else 0.0
    alpha = np.maximum(entry["alpha"], FLOOR)
    return abs(clear_gap(alpha / alpha.sum(), np.asarray(entry["prior"], dtype=float), draws)[1]), 0.0


def _peer(situation: str, role: str) -> str:
    if situation.startswith("rspg_"):
        return f"a typical {role} who is {situation.split('_')[1]}"
    return f"a typical {role} in the same spots"


def _often(share: float) -> str:
    return "usually" if share >= 0.5 else "most often"


def share_takeaway(situation: str, outcomes: list[str], mean: np.ndarray, prior: np.ndarray, draws: np.ndarray, name: str, role: str) -> str:
    lead = situation_lead(situation)
    modal = int(np.argmax(mean))
    usual = act(situation, outcomes[modal])[0]
    best, best_gap = clear_gap(mean, prior, draws)
    if best is None:
        return f"{lead}, {name} {_often(mean[modal])} {usual}, {_pct(mean[modal])}, like {_peer(situation, role)}."
    comparison = f"{_size(best_gap, mean[best], prior[best])}{'more' if best_gap > 0 else 'less'} often than {_peer(situation, role)}: {_pct(mean[best])} against {_pct(prior[best])}"
    if best == modal:
        return f"{lead}, {name} {usual} {comparison}."
    return f"{lead}, {name} {_often(mean[modal])} {usual}, {_pct(mean[modal])}, and {act(situation, outcomes[best])[0]} {comparison}."


def ratio_takeaway(situation: str, mean: float, low: float, high: float, name: str, role: str) -> str:
    words = phrase(situation)
    if low > 1.0 and mean >= 2.0:
        return f"{name} {words}, {mean:.1f} times as often as a typical {role} in the same spots."
    if low > 1.0:
        return f"{name} {words}, {round((mean - 1.0) * 100)}% more often than a typical {role} in the same spots."
    if high < 1.0:
        return f"{name} {words}, {round((1.0 - mean) * 100)}% less often than a typical {role} in the same spots."
    return f"{name} {words} about as often as a typical {role} in the same spots."


def duo_takeaway(situation: str, outcomes: list[str], mean_a: np.ndarray, mean_b: np.ndarray, names: tuple[str, str]) -> str:
    lead = situation_lead(situation)
    left, right = names
    modal_a, modal_b = int(np.argmax(mean_a)), int(np.argmax(mean_b))
    if modal_a != modal_b:
        return (
            f"{lead}, expect {left} to {act(situation, outcomes[modal_a])[1]}, {_pct(mean_a[modal_a])}, "
            f"and {right} to {act(situation, outcomes[modal_b])[1]}, {_pct(mean_b[modal_b])}."
        )
    gaps = np.abs(mean_a - mean_b)
    gaps[modal_a] = -1.0
    k = int(np.argmax(gaps))
    more, high, low = (left, mean_a[k], mean_b[k]) if mean_a[k] > mean_b[k] else (right, mean_b[k], mean_a[k])
    return f"{lead}, you both {_often(min(mean_a[modal_a], mean_b[modal_a]))} {act(situation, outcomes[modal_a])[2]}, but {more} {act(situation, outcomes[k])[0]} more often: {_pct(high)} against {_pct(low)}."


def summarise(entry: dict, rng: np.random.Generator, name: str = "This player", role: str = "player", draws: np.ndarray | None = None) -> dict:
    draws = _draws(entry, rng) if draws is None else draws
    if entry["kind"] == "ratio":
        shape, rate = entry["observed"] + entry["prior"], entry["expected"] + entry["prior"]
        low, high = _band(draws)
        return {
            "situation": entry["situation"],
            "words": entry["words"],
            "kind": "ratio",
            "n": entry["n"],
            "observed": round(entry["observed"], 1),
            "expected": round(entry["expected"], 1),
            "ratio": {"mean": round(shape / rate, 3), "low": round(float(low), 3), "high": round(float(high), 3)},
            "own": round(entry["expected"] / rate, 3),
            "takeaway": ratio_takeaway(entry["situation"], shape / rate, float(low), float(high), name, role),
        }
    alpha = np.maximum(entry["alpha"], FLOOR)
    low, high = _band(draws)
    mean = alpha / alpha.sum()
    return {
        "situation": entry["situation"],
        "words": entry["words"],
        "kind": "shares",
        "n": entry["n"],
        "own": round(float(entry["n"]) / float(alpha.sum()), 3),
        "outcomes": [
            {"name": name_, "words": outcome_label(entry["situation"], name_), "mean": round(float(mean[k]), 4), "low": round(float(low[k]), 4), "high": round(float(high[k]), 4), "prior": round(float(entry["prior"][k]), 4)}
            for k, name_ in enumerate(entry["outcomes"])
            if _shown(name_, float(high[k]), float(entry["prior"][k]))
        ],
        "takeaway": share_takeaway(entry["situation"], list(entry["outcomes"]), mean, np.asarray(entry["prior"], dtype=float), draws, name, role),
    }


def _base(situation: str) -> str:
    parts = situation.split("_")
    return "_".join(["rsp", *parts[2:]]) if parts[0] == "rspg" else situation


def _distinct(items: list, situation) -> list:
    seen, kept = set(), []
    for item in items:
        base = _base(situation(item))
        if base not in seen:
            seen.add(base)
            kept.append(item)
    return kept


def top_situations(entries: list[dict], rng: np.random.Generator, top: int = TOP, name: str = "This player", role: str = "player") -> list[dict]:
    scored = [(entry, _draws(entry, rng)) for entry in entries]
    clear = [(*_clearness(entry, draws), entry, draws) for entry, draws in scored]
    clear = [item for item in clear if item[0] > 0.0 or item[1] > 0.0]
    ranked = _distinct(sorted(clear, key=lambda item: (-item[0], -item[1], -evidence(item[2], rng))), lambda item: item[2]["situation"])[:top]
    return [summarise(entry, rng, name, role, draws) for _, _, entry, draws in ranked]


def differences(left: list[dict], right: list[dict], rng: np.random.Generator, top: int = TOP, names: tuple[str, str] = ("The first player", "the second")) -> list[dict]:
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
    for overlap, a, b, draws_a, draws_b in _distinct(scored, lambda item: item[1]["situation"])[:top]:
        low_a, high_a = _band(draws_a)
        low_b, high_b = _band(draws_b)
        mean_a, mean_b = a["alpha"] / a["alpha"].sum(), b["alpha"] / b["alpha"].sum()
        out.append(
            {
                "situation": a["situation"],
                "words": a["words"],
                "overlap": round(overlap, 3),
                "takeaway": duo_takeaway(a["situation"], list(a["outcomes"]), mean_a, mean_b, names),
                "outcomes": [
                    {
                        "name": name,
                        "words": outcome_label(a["situation"], name),
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


def duo_posteriors(
    left: str, left_position: str, right: str, right_position: str, settings: Settings | None = None, names: tuple[str, str] = ("This player", "this player")
) -> dict:
    settings = settings or get_settings()
    rng = np.random.default_rng(7)
    a, b = player_entries(left, left_position, settings), player_entries(right, right_position, settings)
    left_name, right_name = (short_name(name) for name in names)
    return {
        "left": top_situations(a, rng, name=left_name, role=ROLES.get(left_position, "player")),
        "right": top_situations(b, rng, name=right_name, role=ROLES.get(right_position, "player")),
        "differences": differences(a, b, rng, names=(left_name, right_name)),
        "measured": {"left": bool(a), "right": bool(b)},
    }
