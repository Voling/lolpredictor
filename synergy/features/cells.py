from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from ..ml.movement import best_kappa
from .positions import KEY

FALLBACK_KAPPA = 8.0
MIN_PLAYERS = 50
MIN_GAMES = 5
KAPPA_FLOOR = 3.0
CHUNK_SEATS = 65_536
SINGLE = ["all"]


def _kappa(observed: np.ndarray, expected: np.ndarray) -> float:
    totals = observed.sum(axis=1)
    seen = totals > 0
    if seen.sum() < MIN_PLAYERS:
        return FALLBACK_KAPPA
    return float(best_kappa(observed[seen], totals[seen], expected[seen] / totals[seen, None]))


def expected_counts(chances: np.ndarray, worlds: np.ndarray, position: np.ndarray) -> np.ndarray:
    expected = np.zeros((*chances.shape[:2], worlds.shape[-1]))
    for code in np.unique(position):
        at = position == code
        expected[at] = np.einsum("nst,stk->nsk", chances[at], worlds[:, code])
    return expected


def deviations(others: np.ndarray, worlds: np.ndarray, kappa: np.ndarray, position: np.ndarray) -> np.ndarray:
    chances = others.sum(axis=3)
    expected = expected_counts(chances, worlds, position)
    return (others.sum(axis=2) - expected) / (chances.sum(axis=2) + kappa[None, :])[..., None]


def split_columns(split: dict | None, outcomes: list[str]) -> list[str]:
    if not split:
        return []
    return [f"{split['prefix']}_{group}_{situation}_{outcome}" for group in split["groups"] for situation in split["situations"] for outcome in outcomes]


def readings(others: np.ndarray, worlds: np.ndarray, kappa: np.ndarray, position: np.ndarray, situations: list[str], states: list[str], split: dict | None) -> np.ndarray:
    cells = deviations(others, worlds, kappa, position).reshape(len(others), -1)
    if not split:
        return cells
    picks = [situations.index(situation) for situation in split["situations"]]
    slots = [[states.index(state) for state in group] for group in split["groups"].values()]
    parts = [deviations(others[:, picks][:, :, slot], worlds[picks][:, :, slot], kappa[picks], position) for slot in slots]
    return np.concatenate([cells, np.stack(parts, axis=1).reshape(len(others), -1)], axis=1)


class SeatIndex:
    def __init__(self, seats: pd.DataFrame):
        base = seats[["match_id", *KEY]].drop_duplicates(["match_id", "puuid"])
        self.match_id = base["match_id"].to_numpy()
        self.puuid = base["puuid"].to_numpy()
        self.position = base["position"].to_numpy()
        self.lookup = pd.MultiIndex.from_arrays([self.match_id, self.puuid])
        self.who_code, who = pd.MultiIndex.from_arrays([self.puuid, self.position]).factorize()
        self.who = who.set_names(KEY)
        self.positions = sorted(set(self.who.get_level_values("position")))
        self.who_position = pd.Index(self.positions).get_indexer(self.who.get_level_values("position"))
        self.seat_position = self.who_position[self.who_code]

    def __len__(self) -> int:
        return len(self.match_id)

    def rows(self, match_id, puuid) -> np.ndarray:
        return self.lookup.get_indexer(pd.MultiIndex.from_arrays([np.asarray(match_id), np.asarray(puuid)]))


def moment_kappa(counts: np.ndarray, index: SeatIndex, step: int) -> float:
    width = counts.shape[-1]
    sums, squares, games = np.zeros((len(index.who), width)), np.zeros((len(index.who), width)), np.zeros(len(index.who))
    for start in range(0, len(index), CHUNK_SEATS):
        block = np.asarray(counts[start : start + CHUNK_SEATS, step], dtype=float)
        block = block.reshape(len(block), -1, width).sum(axis=1)
        total = block.sum(axis=1, keepdims=True)
        share = np.where(total > 0, block / np.where(total > 0, total, 1.0), 0.0)
        codes = index.who_code[start : start + CHUNK_SEATS]
        np.add.at(sums, codes, share)
        np.add.at(squares, codes, share**2)
        np.add.at(games, codes, (total[:, 0] > 0).astype(float))
    many = games >= MIN_GAMES
    if many.sum() < MIN_PLAYERS:
        return FALLBACK_KAPPA
    played = games[many, None]
    mean = sums[many] / played
    within = (squares[many] / played - mean**2) * (played / (played - 1.0))
    between = mean.var(axis=0) - (within / played).mean(axis=0)
    pooled_between = float(between.sum())
    if pooled_between <= 0.0:
        return FALLBACK_KAPPA
    return max(float(within.mean(axis=0).sum()) / pooled_between, KAPPA_FLOOR)


@dataclass
class CellFit:
    prefix: str
    situations: list[str]
    states: list[str]
    outcomes: list[str]
    totals: np.ndarray
    worlds: np.ndarray
    kappa: np.ndarray
    report: dict
    unit: float = 1.0
    split: dict | None = None

    @property
    def columns(self) -> list[str]:
        return [f"{self.prefix}_{situation}_{outcome}" for situation in self.situations for outcome in self.outcomes] + split_columns(self.split, self.outcomes)


def dense_counts(
    counts: pd.DataFrame, index: SeatIndex, situations: list[str], outcomes: list[str], out: np.ndarray | None = None, states: list[str] = SINGLE
) -> np.ndarray:
    target = out if out is not None else np.zeros((len(index), len(situations), len(states), len(outcomes)))
    rows = index.rows(counts["match_id"], counts["puuid"])
    situation = pd.Index(situations).get_indexer(counts["situation"])
    state = pd.Index(states).get_indexer(counts["state"]) if len(states) > 1 else np.zeros(len(counts), dtype=np.int64)
    outcome = pd.Index(outcomes).get_indexer(counts["outcome"])
    keep = (rows >= 0) & (situation >= 0) & (state >= 0) & (outcome >= 0)
    code = ((rows[keep].astype(np.int64) * len(situations) + situation[keep]) * len(states) + state[keep]) * len(outcomes) + outcome[keep]
    np.add.at(target.reshape(-1), code, counts["count"].to_numpy(dtype=float)[keep])
    return target


def buffer(path: Path, index: SeatIndex, situations: list[str], outcomes: list[str], states: list[str] = SINGLE) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(index), len(situations), len(states), len(outcomes)))


def fit_cells(
    counts: np.ndarray,
    index: SeatIndex,
    situations: list[str],
    outcomes: list[str],
    prefix: str,
    scale: float = 1.0,
    unit: float = 1.0,
    kappa: float | Sequence[float] | None = None,
    states: list[str] = SINGLE,
    split: dict | None = None,
) -> CellFit:
    shape = (len(situations), len(states), len(outcomes))
    totals = np.zeros((len(index.who), int(np.prod(shape))))
    for start in range(0, len(index), CHUNK_SEATS):
        stop = min(start + CHUNK_SEATS, len(index))
        block = np.asarray(counts[start:stop], dtype=float).reshape(stop - start, -1) / unit
        onehot = sparse.csr_matrix((np.ones(stop - start), (index.who_code[start:stop], np.arange(stop - start))), shape=(len(index.who), stop - start))
        totals += onehot @ block
    totals = totals.reshape(len(index.who), *shape)
    worlds = np.zeros((len(situations), len(index.positions), len(states), len(outcomes)))
    for code in range(len(index.positions)):
        pooled = totals[index.who_position == code].sum(axis=0)
        worlds[:, code] = (pooled + 1.0) / (pooled.sum(axis=2, keepdims=True) + len(outcomes))
    if kappa is None:
        expected = expected_counts(totals.sum(axis=3), worlds, index.who_position)
        found = [_kappa(totals[:, step].sum(axis=1), expected[:, step]) for step in range(len(situations))]
    else:
        found = list(kappa) if isinstance(kappa, Sequence) else [kappa] * len(situations)
    kappa = scale * np.array(found, dtype=float)
    report = {
        situation: {
            "rows": int(totals[:, step].sum()),
            "kappa": round(float(kappa[step]), 2),
            "world": {
                position: {state: [round(float(v), 4) for v in worlds[step, code, at]] for at, state in enumerate(states)}
                for code, position in enumerate(index.positions)
            },
        }
        for step, situation in enumerate(situations)
    }
    return CellFit(prefix, list(situations), list(states), list(outcomes), totals, worlds, kappa, report, unit, split)


def cell_values(fit: CellFit, counts: np.ndarray, index: SeatIndex, start: int, stop: int) -> np.ndarray:
    others = fit.totals[index.who_code[start:stop]]
    others -= np.asarray(counts[start:stop], dtype=float).reshape(others.shape) / fit.unit
    return readings(others, fit.worlds, fit.kappa, index.seat_position[start:stop], fit.situations, fit.states, fit.split)


def _cell_block(parts: list[tuple[CellFit, np.ndarray]], index: SeatIndex, columns: list[str], start: int, stop: int) -> np.ndarray:
    block = np.full((stop - start, len(columns)), np.nan)
    at = {column: position for position, column in enumerate(columns)}
    for fit, counts in parts:
        block[:, [at[column] for column in fit.columns]] = cell_values(fit, counts, index, start, stop)
    return block


def cells_frame(parts: list[tuple[CellFit, np.ndarray]], index: SeatIndex, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(_cell_block(parts, index, columns, 0, len(index)), columns=columns)
    frame.insert(0, "puuid", index.puuid)
    frame.insert(0, "match_id", index.match_id)
    return frame


def write_cells(path: Path, parts: list[tuple[CellFit, np.ndarray]], index: SeatIndex, columns: list[str]) -> int:
    writer = None
    try:
        for start in range(0, len(index), CHUNK_SEATS):
            stop = min(start + CHUNK_SEATS, len(index))
            block = _cell_block(parts, index, columns, start, stop)
            table = pa.table(
                {"match_id": index.match_id[start:stop], "puuid": index.puuid[start:stop], **{column: block[:, k] for k, column in enumerate(columns)}}
            )
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return len(index)


def evidence_shares(fit: CellFit, index: SeatIndex) -> pd.DataFrame:
    exposure = fit.totals.sum(axis=(2, 3))
    kappa = np.array([fit.report[situation]["kappa"] for situation in fit.situations], dtype=float)
    weight = exposure / (exposure + kappa[None, :])
    typical = exposure.mean(axis=0)
    share = (weight * typical[None, :]).sum(axis=1) / max(float(typical.sum()), 1e-9)
    return pd.DataFrame({"puuid": index.who.get_level_values("puuid"), "position": index.who.get_level_values("position"), "share": share})


def combine_shares(parts: list[tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    total = sum(weight for weight, _ in parts)
    merged = None
    for weight, frame in parts:
        scaled = frame.set_index(KEY)["share"] * (weight / total)
        merged = scaled if merged is None else merged.add(scaled, fill_value=0.0)
    return merged.rename("share").reset_index()
