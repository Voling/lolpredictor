import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import Settings, get_settings
from ..features.positions import POSITIONS
from ..ingest.premades import premade_pairs
from ..ingest.store import Store
from .gold import objective_prices
from .interaction import SEED, TEAM_PAIRS, _basis

MINUTE = 15
COMPONENTS = 24
BLOCK = 8000
ADDITIVE_PENALTIES = (1e2, 1e3, 1e4)
PRODUCT_PENALTIES = (1e3, 1e4, 1e5, 1e6, 1e7)
PLANTED = (50.0, 100.0)
NULLS = 3
FLOOR = 10.0
DEPTHS = (20, 50)
SMALLEST_SLICE = 2000
SMALLEST_REFIT = 20000
DEAD_SPREAD = 1e-3
EVENT_VALUES = {"kill": 450.0, "plate": 175.0, "building": 300.0}
TARGETS = ("gold", "events")
SEATS_FILE = "seat_weights.npz"
GOLD_QUERY = (
    "SELECT f.match_id, p.puuid, sum(f.total_gold) AS gold FROM frames f"
    " JOIN participations p ON p.match_id = f.match_id AND p.puuid = f.puuid"
    " WHERE f.minute = %s GROUP BY 1,2"
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


def artifact_names(target: str) -> tuple[str, str]:
    suffix = "" if target == "gold" else f"_{target}"
    return f"interaction_matrix{suffix}.npz", f"interaction_report{suffix}.json"


def seat_gold(settings: Settings, match_id: np.ndarray, seat_puuid: np.ndarray, minute: int = MINUTE) -> np.ndarray:
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


def player_means(basis: dict, rows: np.ndarray) -> dict:
    reduced = basis["reduced"]
    dim = reduced.shape[-1]
    names = pd.Series(basis["seat_puuid"][rows].ravel()) + "|" + pd.Series(basis["seat_position"][rows].ravel())
    codes, keys = pd.factorize(names)
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) > 0])
    counts = np.diff(np.r_[starts, len(order)])
    sums = np.add.reduceat(reduced[rows].reshape(-1, dim)[order].astype(np.float64), starts, axis=0)
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
) -> PairRows:
    additive, products, target, depth, premade = [], [], [], [], []
    combinations = [(a, b) for a in range(len(POSITIONS)) for b in range(a + 1, len(POSITIONS))]
    names, match_id = board.seat_puuid, board.match_id

    def couple(row, one, other):
        first, second = names[row, one], names[row, other]
        return (first, second) if first < second else (second, first)

    for start in range(0, len(rows), BLOCK):
        chunk = rows[start : start + BLOCK]
        cells = board.reduced[chunk].astype(np.float64)
        small = cells @ projection
        at = np.arange(len(chunk))
        for first, second in combinations:
            a, b = board.blue[chunk, first], board.blue[chunk, second]
            c, d = board.red[chunk, first], board.red[chunk, second]
            additive.append((cells[at, a] + cells[at, b]) - (cells[at, c] + cells[at, d]))
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
            order = shuffle.permutation(len(chunk)) if shuffle is not None else at
            ours = small[at, a][:, :, None] * small[order, b[order]][:, None, :]
            theirs = small[at, c][:, :, None] * small[order, d[order]][:, None, :]
            ours, theirs = ours + ours.transpose(0, 2, 1), theirs + theirs.transpose(0, 2, 1)
            products.append((ours - theirs)[:, upper[0], upper[1]])
    return PairRows(
        np.concatenate(additive).astype(np.float32),
        np.concatenate(products).astype(np.float32),
        np.concatenate(target),
        np.concatenate(depth),
        np.concatenate(premade),
    )


def _scaled(fitted: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spread = fitted.std(axis=0, keepdims=True)
    spread[spread == 0.0] = 1.0
    return fitted / spread, held / spread, spread[0]


def _gram(design: np.ndarray, values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((design.shape[1], design.shape[1]))
    right = np.zeros(design.shape[1])
    for start in range(0, len(rows), BLOCK * 4):
        chunk = rows[start : start + BLOCK * 4]
        block = design[chunk]
        left += (block.T @ block).astype(np.float64)
        right += (block.T @ values[chunk].astype(np.float32)).astype(np.float64)
    return left, right


def _fit_pair(left, right, widths, held_design, held_values, middle) -> dict:
    width = widths[0]
    alone = None
    for penalty in ADDITIVE_PENALTIES:
        weights = ridge(left[:width, :width], right[:width], np.repeat(penalty, width))
        score = explained(held_values, middle + held_design[:, :width] @ weights.astype(np.float32))
        if alone is None or score > alone[0]:
            alone = (score, weights)
    together = None
    for first in ADDITIVE_PENALTIES:
        for second in PRODUCT_PENALTIES:
            penalty = np.concatenate([np.repeat(first, widths[0]), np.repeat(second, widths[1])])
            weights = ridge(left, right, penalty)
            score = explained(held_values, middle + held_design @ weights.astype(np.float32))
            if together is None or score > together[0]:
                together = (score, weights)
    return {"alone": alone[0], "alone_weights": alone[1], "together": together[0], "weights": together[1]}


def _slice(name, fitted: PairRows, held: PairRows, fit_design, test_design, mask_fit, mask_test, widths, fit_middle, found) -> dict | None:
    if int(mask_test.sum()) < SMALLEST_SLICE:
        return None
    rows_test = np.nonzero(mask_test)[0]
    values = held.target[rows_test]
    design = test_design[rows_test]
    width = widths[0]
    scored_alone = explained(values, fit_middle + design[:, :width] @ found["alone_weights"].astype(np.float32))
    scored_together = explained(values, fit_middle + design @ found["weights"].astype(np.float32))
    out = {
        "rows_fitted": int(mask_fit.sum()),
        "rows_held_out": int(len(rows_test)),
        "target_spread": round(float(values.std()), 1),
        "scored_gain": round(scored_together - scored_alone, 5),
        "scored_spread": round(float((design[:, width:] @ found["weights"][width:].astype(np.float32)).std()), 1),
    }
    if int(mask_fit.sum()) >= SMALLEST_REFIT:
        rows_fit = np.nonzero(mask_fit)[0]
        middle = float(fitted.target[rows_fit].mean())
        left, right = _gram(fit_design, fitted.target - middle, rows_fit)
        refit = _fit_pair(left, right, widths, design, values, middle)
        out["refit_gain"] = round(refit["together"] - refit["alone"], 5)
        out["refit_spread"] = round(float((design[:, width:] @ refit["weights"][width:].astype(np.float32)).std()), 1)
    print(f"  {name:14} {out}", flush=True)
    return out


def fit_mirrored(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    components: int = COMPONENTS,
    seed: int = SEED,
    source: str = "style",
    target: str = "gold",
) -> dict:
    settings = settings or get_settings()
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
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
        f" {len(board.premade):,} premade pairs known, target {target}",
        flush=True,
    )

    projection = directions(reduced, fit, components)
    upper = np.triu_indices(components)
    fitted = pair_design(board, projection, fit, upper)
    held = pair_design(board, projection, test, upper)
    fitted.additive, held.additive, _ = _scaled(fitted.additive, held.additive)
    fitted.products, held.products, product_spread = _scaled(fitted.products, held.products)
    widths = (fitted.additive.shape[1], fitted.products.shape[1])
    fit_design = np.hstack([fitted.additive, fitted.products])
    test_design = np.hstack([held.additive, held.products])
    print(f"{len(fitted.target):,} pair rows fitted, spread {fitted.target.std():,.0f} gold", flush=True)

    middle = float(fitted.target.mean())
    left, right = _gram(fit_design, fitted.target - middle, np.arange(len(fitted.target)))
    found = _fit_pair(left, right, widths, test_design, held.target, middle)
    weights = found["weights"]
    product_weights = weights[widths[0] :] / product_spread
    compact = np.zeros((components, components))
    compact[upper[0], upper[1]] = product_weights
    compact = (compact + compact.T) / 2.0
    matrix = projection @ compact @ projection.T * (TEAM_PAIRS * dim)
    own = held.products @ weights[widths[0] :].astype(np.float32)
    report = {
        "target": (
            "each pair's gold at 15 against the same two enemy seats"
            if target == "gold"
            else "gold swung by kills, plates and objectives both players were present at, against the same two enemy seats"
        ),
        "matches": int(sound.sum()),
        "pair_rows": int(len(fitted.target)),
        "held_out_rows": int(len(held.target)),
        "columns": dim,
        "components": components,
        "target_spread": round(float(fitted.target.std()), 1),
        "cells_alone": round(found["alone"], 5),
        "with_interaction": round(found["together"], 5),
        "gain": round(found["together"] - found["alone"], 5),
        "interaction_spread": round(float(own.std()), 1),
    }
    print(
        f"cells alone {report['cells_alone']:+.5f}, with the interaction {report['with_interaction']:+.5f},"
        f" spread {report['interaction_spread']} gold",
        flush=True,
    )

    slices = {}
    for depth in DEPTHS:
        slices[f"both pairs {depth}+ games in position"] = (fitted.depth >= depth, held.depth >= depth)
    slices["premade"] = (fitted.premade, held.premade)
    slices["strangers"] = (~fitted.premade, ~held.premade)
    report["slices"] = {}
    for name, (mask_fit, mask_test) in slices.items():
        result = _slice(name, fitted, held, fit_design, test_design, mask_fit, mask_test, widths, middle, found)
        if result is not None:
            report["slices"][name] = result

    rng = np.random.default_rng(1000)
    draws = []
    for draw in range(NULLS):
        shuffled_fit = pair_design(board, projection, fit, upper, shuffle=rng).products
        shuffled_test = pair_design(board, projection, test, upper, shuffle=rng).products
        shuffled_fit, shuffled_test, _ = _scaled(shuffled_fit, shuffled_test)
        null_design = np.hstack([fitted.additive, shuffled_fit])
        null_held = np.hstack([held.additive, shuffled_test])
        null_left, null_right = _gram(null_design, fitted.target - middle, np.arange(len(fitted.target)))
        null = _fit_pair(null_left, null_right, widths, null_held, held.target, middle)
        draws.append(round(null["together"] - null["alone"], 5))
        print(f"  partners shuffled {draw + 1}/{NULLS}: the interaction adds {draws[-1]:+.5f}", flush=True)
    report["shuffled"] = draws

    planted = {}
    for level in PLANTED:
        seeded = np.random.default_rng(7).normal(size=(components, components))
        seeded = (seeded + seeded.T)[upper[0], upper[1]].astype(np.float32)
        raw_fit, raw_test = fitted.products @ seeded, held.products @ seeded
        scale = level / float(raw_fit.std())
        planted_fit, planted_test = fitted.target + raw_fit * scale, held.target + raw_test * scale
        planted_middle = float(planted_fit.mean())
        planted_left, planted_right = _gram(fit_design, planted_fit - planted_middle, np.arange(len(planted_fit)))
        recovered = _fit_pair(planted_left, planted_right, widths, test_design, planted_test, planted_middle)
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
        units=np.array("gold at 15"),
    )
    (settings.model_dir / report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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
    held, weights = {}, np.zeros((len(POSITIONS), len(columns)))
    centres, quantiles = np.zeros((len(POSITIONS), len(columns))), np.zeros((len(POSITIONS), len(grid)))
    scale = np.ones(len(POSITIONS))
    for index, name in enumerate(POSITIONS):
        design_fit = reduced[fit, blue[fit, index]].astype(np.float32) - reduced[fit, red[fit, index]].astype(np.float32)
        design_test = reduced[test, blue[test, index]].astype(np.float32) - reduced[test, red[test, index]].astype(np.float32)
        values_fit = gold[fit, blue[fit, index]] - gold[fit, red[fit, index]]
        values_test = gold[test, blue[test, index]] - gold[test, red[test, index]]
        spread = design_fit.std(axis=0)
        dead = spread < DEAD_SPREAD
        spread[dead] = 1.0
        design_fit[:, dead] = 0.0
        design_test[:, dead] = 0.0
        design_fit, design_test = design_fit / spread, design_test / spread
        middle = float(values_fit.mean())
        left, right = _gram(design_fit, values_fit - middle, np.arange(len(values_fit)))
        chosen = None
        for penalty in ADDITIVE_PENALTIES:
            found = ridge(left, right, np.repeat(penalty, design_fit.shape[1]))
            score = explained(values_test, middle + design_test @ found.astype(np.float32))
            if chosen is None or score > chosen[0]:
                chosen = (score, found / spread)
        held[name] = round(chosen[0], 5)
        weights[index] = np.where(dead, 0.0, chosen[1])
        seats = np.concatenate([reduced[fit, blue[fit, index]], reduced[fit, red[fit, index]]]).astype(np.float64)
        centres[index] = seats.mean(axis=0)

        def served(rows, side):
            picked = [table["at"][f"{names[row, seat]}|{name}"] for row, seat in zip(rows, side[rows, index])]
            return (table["means"][picked] - centres[index]) @ weights[index]

        scale[index] = calibration(served(test, blue) - served(test, red), values_test)
        readings = scale[index] * np.concatenate([served(fit, blue), served(fit, red)])
        quantiles[index] = np.quantile(readings, grid)
        print(
            f"  {name:8} seat edge at 15, spread {values_fit.std():,.0f} gold, held out r2 {chosen[0]:+.4f},"
            f" {int(dead.sum())} dead cells, served readings scaled by {scale[index]:.3f},"
            f" spread {readings.std():,.0f} gold over {len(readings):,} seats",
            flush=True,
        )
    np.savez(
        settings.model_dir / SEATS_FILE,
        weights=weights,
        centres=centres,
        scale=scale,
        quantiles=quantiles,
        columns=np.array(columns),
        positions=np.array(list(POSITIONS)),
    )
    return {"held_out": held, "matches": int(sound.sum()), "columns": len(columns)}
