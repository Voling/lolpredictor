import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse

from ..config import Settings, get_settings
from ..deep.stream import SEATS
from ..features.positions import POSITIONS
from ..ingest.premades import premade_pairs
from ..ingest.store import Store
from .gold import objective_prices
from .interaction import SEED, TEAM_PAIRS, _basis

COMPONENTS = 24
BLOCK = 8000
ADDITIVE_PENALTIES = (1e2, 1e3, 1e4)
SEAT_PENALTIES = (1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5)
PRODUCT_PENALTIES = (1e3, 1e4, 1e5, 1e6, 1e7)
PLANTED = (50.0, 100.0)
NULLS = 3
FLOOR = 10.0
DEPTHS = (20, 50)
SMALLEST_SLICE = 2000
SMALLEST_REFIT = 20000
DEAD_SPREAD = 1e-3
VALIDATION_SHARE = 0.125
EVENT_VALUES = {"kill": 450.0, "plate": 175.0, "building": 300.0}
TARGETS = ("gold", "events")
SEATS_FILE = "seat_weights.npz"
SEAT_REPORT_FILE = "seat_report.json"
GOLD_QUERY = (
    "SELECT f.match_id, p.puuid, sum(f.total_gold) AS gold FROM frames f"
    " JOIN participations p ON p.match_id = f.match_id AND p.puuid = f.puuid"
    " JOIN (SELECT match_id, max(minute) AS last FROM frames GROUP BY match_id) l ON l.match_id = f.match_id"
    " WHERE f.minute = LEAST(%s, l.last) GROUP BY 1,2"
)


@dataclass
class Board:
    reduced: np.ndarray
    gold: np.ndarray
    blue: np.ndarray
    red: np.ndarray
    games: np.ndarray
    match_id: np.ndarray
    seat_puuid: np.ndarray
    premade: set
    swings: dict | None


@dataclass
class PairRows:
    additive: np.ndarray
    products: np.ndarray
    target: np.ndarray
    depth: np.ndarray
    premade: np.ndarray


def on_device(array: np.ndarray, device: str) -> torch.Tensor:
    out = torch.empty(array.shape, dtype=torch.float32, device=device)
    for start in range(0, len(array), BLOCK):
        out[start : start + BLOCK] = torch.from_numpy(np.array(array[start : start + BLOCK], dtype=np.float32)).to(device)
    return out


def artifact_names(target: str) -> tuple[str, str]:
    suffix = "" if target == "gold" else f"_{target}"
    return f"interaction_matrix{suffix}.npz", f"interaction_report{suffix}.json"


def seat_gold(settings: Settings, match_id: np.ndarray, seat_puuid: np.ndarray, minute: int | None = None) -> np.ndarray:
    minute = minute or settings.target_minute
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute(GOLD_QUERY, (minute,))
            rows = pd.DataFrame(cursor.fetchall())
    finally:
        store.close()
    known = {(row.match_id, row.puuid): float(row.gold) for row in rows.itertuples()}
    gold = np.full(seat_puuid.shape, np.nan)
    for row in range(len(match_id)):
        for seat in range(seat_puuid.shape[1]):
            found = known.get((match_id[row], seat_puuid[row, seat]))
            if found is not None:
                gold[row, seat] = found
    return gold


def seats_by_position(positions: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    place = {name: index for index, name in enumerate(POSITIONS)}
    blue = np.full((len(positions), len(POSITIONS)), -1, dtype=np.int32)
    red = np.full((len(positions), len(POSITIONS)), -1, dtype=np.int32)
    for row in range(len(positions)):
        for seat in range(positions.shape[1]):
            found = place.get(str(positions[row, seat]))
            if found is None:
                continue
            target = blue if side[row, seat] == 1 else red
            if target[row, found] < 0:
                target[row, found] = seat
    return blue, red


def seat_games(seat_puuid: np.ndarray, seat_position: np.ndarray) -> np.ndarray:
    keys = pd.Series(seat_puuid.ravel().astype(str)) + "|" + pd.Series(seat_position.ravel().astype(str))
    codes = pd.factorize(keys)[0]
    return np.bincount(codes)[codes].reshape(seat_puuid.shape)


def pair_swings(frame: pd.DataFrame, prices: dict) -> dict:
    value = frame.trigger.astype(str).map(EVENT_VALUES).astype(float)
    objective = (frame.trigger.astype(str) == "objective").to_numpy()
    value = value.to_numpy()
    value[objective] = frame.detail.astype(str)[objective].map(prices).fillna(0.0).to_numpy()
    keyed = pd.DataFrame(
        {
            "match": frame.match_id.astype(str).to_numpy(),
            "trigger": frame.trigger_id.to_numpy(),
            "team": frame.team_id.to_numpy(),
            "puuid": frame.puuid.astype(str).to_numpy(),
            "signed": np.where(frame.ours.to_numpy() == 1, value, -value),
        }
    )
    pairs = keyed.merge(keyed[["match", "trigger", "team", "puuid"]], on=["match", "trigger", "team"])
    pairs = pairs[pairs.puuid_x < pairs.puuid_y]
    summed = pairs.groupby(["match", "puuid_x", "puuid_y"])["signed"].sum()
    return {key: float(total) for key, total in summed.items()}


def joint_swings(settings: Settings) -> dict:
    table = pq.read_table(
        settings.processed_dir / "event_responses.parquet",
        columns=["match_id", "trigger_id", "trigger", "detail", "puuid", "team_id", "ours", "present"],
        filters=[("present", "==", 1.0)],
    )
    frame = table.to_pandas(strings_to_categorical=True)
    print(f"joint events: {len(frame):,} present rows", flush=True)
    return pair_swings(frame, objective_prices(settings))


def validation_split(rows: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed + 1).permutation(len(rows))
    cut = int(len(rows) * VALIDATION_SHARE)
    return np.sort(rows[order[cut:]]), np.sort(rows[order[:cut]])


def player_means(basis: dict, rows: np.ndarray) -> dict:
    reduced = basis["reduced"]
    dim = reduced.shape[-1]
    names = pd.Series(basis["seat_puuid"][rows].ravel()) + "|" + pd.Series(basis["seat_position"][rows].ravel())
    codes, keys = pd.factorize(names)
    counts = np.bincount(codes, minlength=len(keys))
    sums = np.zeros((len(keys), dim))
    for start in range(0, len(rows), BLOCK):
        picked = codes[start * SEATS : (start + BLOCK) * SEATS]
        block = np.asarray(reduced[rows[start : start + BLOCK]], dtype=np.float64).reshape(-1, dim)
        sums += sparse.csr_matrix((np.ones(len(picked)), (picked, np.arange(len(picked)))), shape=(len(keys), len(picked))) @ block
    return {
        "means": (sums / counts[:, None]).astype(np.float32),
        "positions": np.array([str(key).split("|")[1] for key in keys]),
        "counts": counts,
        "at": {str(key): index for index, key in enumerate(keys)},
    }


def calibration(predicted: np.ndarray, realised: np.ndarray) -> float:
    centred = predicted - predicted.mean()
    spread = float((centred**2).sum())
    return float((centred * (realised - realised.mean())).sum() / spread) if spread > 0.0 else 1.0


def directions(reduced: np.ndarray, rows: np.ndarray, components: int) -> np.ndarray:
    sample = reduced[rows[: min(len(rows), 20000)]].reshape(-1, reduced.shape[-1]).astype(np.float64)
    sample = sample - sample.mean(axis=0)
    _, _, found = np.linalg.svd(sample[:: max(1, len(sample) // 40000)], full_matrices=False)
    return found[:components].T


def ridge(left: np.ndarray, right: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return np.linalg.solve(left + np.diag(penalty), right)


def explained(values: np.ndarray, guess: np.ndarray) -> float:
    return 1.0 - float(((values - guess) ** 2).sum()) / float(((values - values.mean()) ** 2).sum())


def pair_design(
    board: Board,
    projection: np.ndarray,
    rows: np.ndarray,
    upper: tuple[np.ndarray, np.ndarray],
    shuffle: np.random.Generator | None = None,
    cells: torch.Tensor | None = None,
) -> PairRows:
    source = cells if cells is not None else torch.as_tensor(board.reduced)
    device = source.device
    project = torch.as_tensor(projection, dtype=torch.float32, device=device)
    left_index = torch.as_tensor(upper[0], device=device)
    right_index = torch.as_tensor(upper[1], device=device)
    additive, products, target, depth, premade = [], [], [], [], []
    combinations = [(a, b) for a in range(len(POSITIONS)) for b in range(a + 1, len(POSITIONS))]
    names, match_id = board.seat_puuid, board.match_id

    def couple(row, one, other):
        first, second = names[row, one], names[row, other]
        return (first, second) if first < second else (second, first)

    def seat(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    for start in range(0, len(rows), BLOCK):
        chunk = rows[start : start + BLOCK]
        block = source[seat(chunk)].float()
        small = block @ project
        at = torch.arange(len(chunk), device=device)
        for first, second in combinations:
            a, b = board.blue[chunk, first], board.blue[chunk, second]
            c, d = board.red[chunk, first], board.red[chunk, second]
            ta, tb, tc, td = seat(a), seat(b), seat(c), seat(d)
            additive.append((block[at, ta] + block[at, tb]) - (block[at, tc] + block[at, td]))
            if board.swings is None:
                target.append(board.gold[chunk, a] + board.gold[chunk, b] - board.gold[chunk, c] - board.gold[chunk, d])
            else:
                target.append(
                    np.array(
                        [
                            board.swings.get((match_id[row], *couple(row, a[k], b[k])), 0.0)
                            - board.swings.get((match_id[row], *couple(row, c[k], d[k])), 0.0)
                            for k, row in enumerate(chunk)
                        ]
                    )
                )
            depth.append(np.minimum.reduce([board.games[chunk, a], board.games[chunk, b], board.games[chunk, c], board.games[chunk, d]]))
            premade.append(np.array([couple(row, a[k], b[k]) in board.premade for k, row in enumerate(chunk)]))
            order = seat(shuffle.permutation(len(chunk))) if shuffle is not None else at
            ours = small[at, ta][:, :, None] * small[order, tb[order]][:, None, :]
            theirs = small[at, tc][:, :, None] * small[order, td[order]][:, None, :]
            ours, theirs = ours + ours.transpose(1, 2), theirs + theirs.transpose(1, 2)
            products.append((ours - theirs)[:, left_index, right_index])
    return PairRows(
        torch.cat(additive),
        torch.cat(products),
        np.concatenate(target),
        np.concatenate(depth),
        np.concatenate(premade),
    )


def _scaled(fitted: torch.Tensor, held: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    spread = fitted.std(dim=0, unbiased=False, keepdim=True)
    spread[spread == 0.0] = 1.0
    return fitted / spread, held / spread, spread[0].cpu().numpy().astype(np.float64)


def _gram(design: np.ndarray, values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((design.shape[1], design.shape[1]))
    right = np.zeros(design.shape[1])
    for start in range(0, len(rows), BLOCK * 4):
        chunk = rows[start : start + BLOCK * 4]
        block = design[chunk]
        left += (block.T @ block).astype(np.float64)
        right += (block.T @ values[chunk].astype(np.float32)).astype(np.float64)
    return left, right


def _gram_parts(parts: list[torch.Tensor], values: torch.Tensor, rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    width = sum(part.shape[1] for part in parts)
    device = parts[0].device
    left = torch.zeros((width, width), dtype=torch.float64, device=device)
    right = torch.zeros(width, dtype=torch.float64, device=device)
    count = len(rows) if rows is not None else len(values)
    for start in range(0, count, BLOCK * 4):
        pick = rows[start : start + BLOCK * 4] if rows is not None else slice(start, start + BLOCK * 4)
        block = torch.cat([part[pick] for part in parts], dim=1)
        left += (block.T @ block).double()
        right += (block.T @ values[pick].float()).double()
    return left, right


def _solve(left: torch.Tensor, right: torch.Tensor, penalty: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(left + torch.diag(penalty), right)


def _explained(values: torch.Tensor, guess: torch.Tensor) -> float:
    values, guess = values.double(), guess.double()
    return 1.0 - float(((values - guess) ** 2).sum()) / float(((values - values.mean()) ** 2).sum())


def _predict(parts: list[torch.Tensor], weights: torch.Tensor, middle: float) -> torch.Tensor:
    out, start = torch.full((parts[0].shape[0],), middle, dtype=torch.float32, device=parts[0].device), 0
    for part in parts:
        out += part @ weights[start : start + part.shape[1]].float()
        start += part.shape[1]
    return out


def _fit_pair(left, right, widths, held_parts, held_values, middle) -> dict:
    width = widths[0]
    device = left.device
    alone = None
    for penalty in ADDITIVE_PENALTIES:
        weights = _solve(left[:width, :width], right[:width], torch.full((width,), penalty, dtype=torch.float64, device=device))
        score = _explained(held_values, _predict(held_parts[:1], weights, middle))
        if alone is None or score > alone[0]:
            alone = (score, weights)
    together = None
    for first in ADDITIVE_PENALTIES:
        for second in PRODUCT_PENALTIES:
            penalty = torch.cat(
                [torch.full((widths[0],), first, dtype=torch.float64, device=device), torch.full((widths[1],), second, dtype=torch.float64, device=device)]
            )
            weights = _solve(left, right, penalty)
            score = _explained(held_values, _predict(held_parts, weights, middle))
            if together is None or score > together[0]:
                together = (score, weights)
    return {"alone": alone[0], "alone_weights": alone[1], "together": together[0], "weights": together[1]}


def _slice(name, fitted: PairRows, held: PairRows, fit_target, test_target, mask_fit, mask_test, widths, fit_middle, found) -> dict | None:
    if int(mask_test.sum()) < SMALLEST_SLICE:
        return None
    device = fit_target.device
    rows_test = torch.as_tensor(np.nonzero(mask_test)[0], device=device)
    values = test_target[rows_test]
    parts = [held.additive[rows_test], held.products[rows_test]]
    width = widths[0]
    scored_alone = _explained(values, _predict(parts[:1], found["alone_weights"], fit_middle))
    scored_together = _explained(values, _predict(parts, found["weights"], fit_middle))
    out = {
        "rows_fitted": int(mask_fit.sum()),
        "rows_held_out": int(len(rows_test)),
        "target_spread": round(float(values.double().std(unbiased=False)), 1),
        "scored_gain": round(scored_together - scored_alone, 5),
        "scored_spread": round(float((parts[1] @ found["weights"][width:].float()).double().std(unbiased=False)), 1),
    }
    if int(mask_fit.sum()) >= SMALLEST_REFIT:
        rows_fit = torch.as_tensor(np.nonzero(mask_fit)[0], device=device)
        middle = float(fit_target[rows_fit].double().mean())
        left, right = _gram_parts([fitted.additive, fitted.products], fit_target - middle, rows_fit)
        refit = _fit_pair(left, right, widths, parts, values, middle)
        out["refit_gain"] = round(refit["together"] - refit["alone"], 5)
        out["refit_spread"] = round(float((parts[1] @ refit["weights"][width:].float()).double().std(unbiased=False)), 1)
    print(f"  {name:14} {out}", flush=True)
    return out


def fit_mirrored(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    components: int = COMPONENTS,
    seed: int = SEED,
    source: str = "style",
    target: str = "gold",
    device: str | None = None,
) -> dict:
    settings = settings or get_settings()
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clock = time.time()
    basis = _basis(settings, stream, seed, source)
    reduced, columns = basis["reduced"], [str(name) for name in basis["columns"]]
    dim = len(columns)
    blue, red = seats_by_position(basis["seat_position"], basis["seat_side"])
    gold = seat_gold(settings, basis["match_id"], basis["seat_puuid"])
    board = Board(
        reduced=reduced,
        gold=gold,
        blue=blue,
        red=red,
        games=seat_games(basis["seat_puuid"], basis["seat_position"]),
        match_id=basis["match_id"],
        seat_puuid=basis["seat_puuid"],
        premade=premade_pairs(settings),
        swings=joint_swings(settings) if target == "events" else None,
    )
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    fit = np.array(sorted(set(basis["fit"]) & set(np.nonzero(sound)[0])))
    test = np.array(sorted(set(basis["test"]) & set(np.nonzero(sound)[0])))
    print(
        f"mirrored pairs on {int(sound.sum()):,} matches, fitting {len(fit):,}, holding out {len(test):,},"
        f" {len(board.premade):,} premade pairs known, target {target}, on {device}, basis ready in {time.time() - clock:.0f}s",
        flush=True,
    )

    clock = time.time()
    cells = on_device(reduced, device)
    projection = directions(reduced, fit, components)
    upper = np.triu_indices(components)
    fitted = pair_design(board, projection, fit, upper, cells=cells)
    held = pair_design(board, projection, test, upper, cells=cells)
    fitted.additive, held.additive, _ = _scaled(fitted.additive, held.additive)
    fitted.products, held.products, product_spread = _scaled(fitted.products, held.products)
    widths = (fitted.additive.shape[1], fitted.products.shape[1])
    fit_target = torch.as_tensor(fitted.target, dtype=torch.float32, device=device)
    test_target = torch.as_tensor(held.target, dtype=torch.float32, device=device)
    held_parts = [held.additive, held.products]
    print(f"{len(fitted.target):,} pair rows fitted, spread {fitted.target.std():,.0f} gold, design built in {time.time() - clock:.0f}s", flush=True)

    clock = time.time()
    middle = float(fitted.target.mean())
    left, right = _gram_parts([fitted.additive, fitted.products], fit_target - middle)
    found = _fit_pair(left, right, widths, held_parts, test_target, middle)
    weights = found["weights"].cpu().numpy()
    product_weights = weights[widths[0] :] / product_spread
    compact = np.zeros((components, components))
    compact[upper[0], upper[1]] = product_weights
    compact = (compact + compact.T) / 2.0
    matrix = projection @ compact @ projection.T * (TEAM_PAIRS * dim)
    own = held.products @ found["weights"][widths[0] :].float()
    report = {
        "target": (
            f"each pair's gold at {settings.target_minute} against the same two enemy seats"
            if target == "gold"
            else "gold swung by kills, plates and objectives both players were present at, against the same two enemy seats"
        ),
        "device": device,
        "minute": settings.target_minute,
        "matches": int(sound.sum()),
        "pair_rows": int(len(fitted.target)),
        "held_out_rows": int(len(held.target)),
        "columns": dim,
        "components": components,
        "target_spread": round(float(fitted.target.std()), 1),
        "cells_alone": round(found["alone"], 5),
        "with_interaction": round(found["together"], 5),
        "gain": round(found["together"] - found["alone"], 5),
        "interaction_spread": round(float(own.double().std(unbiased=False)), 1),
    }
    print(
        f"cells alone {report['cells_alone']:+.5f}, with the interaction {report['with_interaction']:+.5f},"
        f" spread {report['interaction_spread']} gold, fitted in {time.time() - clock:.0f}s",
        flush=True,
    )

    slices = {}
    for depth in DEPTHS:
        slices[f"both pairs {depth}+ games in position"] = (fitted.depth >= depth, held.depth >= depth)
    slices["premade"] = (fitted.premade, held.premade)
    slices["strangers"] = (~fitted.premade, ~held.premade)
    report["slices"] = {}
    for name, (mask_fit, mask_test) in slices.items():
        result = _slice(name, fitted, held, fit_target, test_target, mask_fit, mask_test, widths, middle, found)
        if result is not None:
            report["slices"][name] = result

    rng = np.random.default_rng(1000)
    draws = []
    for draw in range(NULLS):
        shuffled_fit = pair_design(board, projection, fit, upper, shuffle=rng, cells=cells).products
        shuffled_test = pair_design(board, projection, test, upper, shuffle=rng, cells=cells).products
        shuffled_fit, shuffled_test, _ = _scaled(shuffled_fit, shuffled_test)
        null_left, null_right = _gram_parts([fitted.additive, shuffled_fit], fit_target - middle)
        null = _fit_pair(null_left, null_right, widths, [held.additive, shuffled_test], test_target, middle)
        del shuffled_fit, shuffled_test
        draws.append(round(null["together"] - null["alone"], 5))
        print(f"  partners shuffled {draw + 1}/{NULLS}: the interaction adds {draws[-1]:+.5f}", flush=True)
    report["shuffled"] = draws

    planted = {}
    for level in PLANTED:
        seeded = np.random.default_rng(7).normal(size=(components, components))
        seeded = torch.as_tensor((seeded + seeded.T)[upper[0], upper[1]], dtype=torch.float32, device=device)
        raw_fit, raw_test = fitted.products @ seeded, held.products @ seeded
        scale = level / float(raw_fit.double().std(unbiased=False))
        planted_fit, planted_test = fit_target + raw_fit * scale, test_target + raw_test * scale
        planted_middle = float(planted_fit.double().mean())
        planted_left, planted_right = _gram_parts([fitted.additive, fitted.products], planted_fit - planted_middle)
        recovered = _fit_pair(planted_left, planted_right, widths, held_parts, planted_test, planted_middle)
        planted[f"{level:.0f} gold"] = round(recovered["together"] - recovered["alone"], 5)
        print(f"  planted {level:.0f} gold: the interaction adds {planted[f'{level:.0f} gold']:+.5f}", flush=True)
    report["planted"] = planted
    report["informative"] = bool(
        report["interaction_spread"] >= FLOOR and report["gain"] > max(draws) and planted[f"{PLANTED[0]:.0f} gold"] > 0.0
    )

    matrix_file, report_file = artifact_names(target)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        settings.model_dir / matrix_file,
        matrix=matrix,
        columns=np.array(columns),
        centre=basis["centre"],
        spread=basis["spread"],
        units=np.array(f"gold at {settings.target_minute}"),
        minute=settings.target_minute,
    )
    (settings.model_dir / report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def chosen_ridge(core: tuple[np.ndarray, np.ndarray], check: tuple[np.ndarray, np.ndarray], penalties: tuple = SEAT_PENALTIES) -> dict:
    design, values = core
    design_check, values_check = check
    middle = float(values.mean())
    left, right = _gram(design, values - middle, np.arange(len(values)))
    chosen = None
    for penalty in penalties:
        found = ridge(left, right, np.repeat(penalty, design.shape[1]))
        score = explained(values_check, middle + design_check @ found.astype(np.float32))
        if chosen is None or score > chosen[0]:
            chosen = (score, penalty, found)
    pooled = float((values.sum() + values_check.sum()) / (len(values) + len(values_check)))
    extra_left, extra_right = _gram(design_check, values_check - pooled, np.arange(len(values_check)))
    shifted = right - (pooled - middle) * design.sum(axis=0, dtype=np.float64)
    weights = ridge(left + extra_left, shifted + extra_right, np.repeat(chosen[1], design.shape[1]))
    return {"validation": chosen[0], "penalty": chosen[1], "core": chosen[2], "weights": weights, "middle": pooled}


def fit_positions(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    seed: int = SEED,
    source: str = "style",
) -> dict:
    settings = settings or get_settings()
    basis = _basis(settings, stream, seed, source)
    reduced, columns = basis["reduced"], [str(name) for name in basis["columns"]]
    blue, red = seats_by_position(basis["seat_position"], basis["seat_side"])
    gold = seat_gold(settings, basis["match_id"], basis["seat_puuid"])
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    fit = np.array(sorted(set(basis["fit"]) & set(np.nonzero(sound)[0])))
    test = np.array(sorted(set(basis["test"]) & set(np.nonzero(sound)[0])))
    grid = np.linspace(0.0, 1.0, 1001)
    table = player_means(basis, np.array(sorted(set(fit) | set(test))))
    names = basis["seat_puuid"]
    core, check = validation_split(fit, seed)
    held, validated, penalties, weights = {}, {}, {}, np.zeros((len(POSITIONS), len(columns)))
    centres, quantiles = np.zeros((len(POSITIONS), len(columns))), np.zeros((len(POSITIONS), len(grid)))
    scale = np.ones(len(POSITIONS))
    for index, name in enumerate(POSITIONS):
        design_fit = reduced[core, blue[core, index]].astype(np.float32) - reduced[core, red[core, index]].astype(np.float32)
        design_check = reduced[check, blue[check, index]].astype(np.float32) - reduced[check, red[check, index]].astype(np.float32)
        design_test = reduced[test, blue[test, index]].astype(np.float32) - reduced[test, red[test, index]].astype(np.float32)
        values_fit = gold[core, blue[core, index]] - gold[core, red[core, index]]
        values_check = gold[check, blue[check, index]] - gold[check, red[check, index]]
        values_test = gold[test, blue[test, index]] - gold[test, red[test, index]]
        spread = design_fit.std(axis=0)
        dead = spread < DEAD_SPREAD
        spread[dead] = 1.0
        for design in (design_fit, design_check, design_test):
            design[:, dead] = 0.0
            design /= spread
        found = chosen_ridge((design_fit, values_fit), (design_check, values_check))
        validated[name], penalties[name] = round(found["validation"], 5), found["penalty"]
        held[name] = round(explained(values_test, found["middle"] + design_test @ found["weights"].astype(np.float32)), 5)
        weights[index] = np.where(dead, 0.0, found["weights"] / spread)
        checked = np.where(dead, 0.0, found["core"] / spread)
        centres[index] = (reduced[fit, blue[fit, index]].mean(axis=0, dtype=np.float64) + reduced[fit, red[fit, index]].mean(axis=0, dtype=np.float64)) / 2.0

        def served(rows, side, using):
            picked = [table["at"][f"{names[row, seat]}|{name}"] for row, seat in zip(rows, side[rows, index])]
            return (table["means"][picked] - centres[index]) @ using

        scale[index] = calibration(served(check, blue, checked) - served(check, red, checked), values_check)
        readings = scale[index] * np.concatenate([served(fit, blue, weights[index]), served(fit, red, weights[index])])
        quantiles[index] = np.quantile(readings, grid)
        print(
            f"  {name:8} seat edge at {settings.target_minute}, spread {values_fit.std():,.0f} gold, penalty {found['penalty']:,.0f},"
            f" validation r2 {found['validation']:+.4f}, held out r2 {held[name]:+.4f}, {int(dead.sum())} dead cells,"
            f" served readings scaled by {scale[index]:.3f}, spread {readings.std():,.0f} gold over {len(readings):,} seats",
            flush=True,
        )
    np.savez(
        settings.model_dir / SEATS_FILE,
        weights=weights,
        centres=centres,
        scale=scale,
        quantiles=quantiles,
        minute=settings.target_minute,
        columns=np.array(columns),
        positions=np.array(list(POSITIONS)),
    )
    report = {
        "minute": settings.target_minute,
        "held_out": held,
        "validation": validated,
        "penalty": penalties,
        "scale": {name: round(float(scale[index]), 3) for index, name in enumerate(POSITIONS)},
        "matches": int(sound.sum()),
        "columns": len(columns),
    }
    (settings.model_dir / SEAT_REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
