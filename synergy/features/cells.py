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


def _kappa(counts: np.ndarray, world: np.ndarray) -> float:
    totals = counts.sum(axis=1)
    seen = totals > 0
    if seen.sum() < MIN_PLAYERS:
        return FALLBACK_KAPPA
    return float(best_kappa(counts[seen], totals[seen], world[seen]))


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
    width = counts.shape[2]
    sums, squares, games = np.zeros((len(index.who), width)), np.zeros((len(index.who), width)), np.zeros(len(index.who))
    for start in range(0, len(index), CHUNK_SEATS):
        block = np.asarray(counts[start : start + CHUNK_SEATS, step], dtype=float)
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
    outcomes: list[str]
    totals: np.ndarray
    worlds: np.ndarray
    kappa: np.ndarray
    report: dict
    unit: float = 1.0

    @property
    def columns(self) -> list[str]:
        return [f"{self.prefix}_{situation}_{outcome}" for situation in self.situations for outcome in self.outcomes]


def dense_counts(counts: pd.DataFrame, index: SeatIndex, situations: list[str], outcomes: list[str], out: np.ndarray | None = None) -> np.ndarray:
    target = out if out is not None else np.zeros((len(index), len(situations), len(outcomes)))
    rows = index.rows(counts["match_id"], counts["puuid"])
    situation = pd.Index(situations).get_indexer(counts["situation"])
    outcome = pd.Index(outcomes).get_indexer(counts["outcome"])
    keep = (rows >= 0) & (situation >= 0) & (outcome >= 0)
    code = (rows[keep].astype(np.int64) * len(situations) + situation[keep]) * len(outcomes) + outcome[keep]
    np.add.at(target.reshape(-1), code, counts["count"].to_numpy(dtype=float)[keep])
    return target


def buffer(path: Path, index: SeatIndex, situations: list[str], outcomes: list[str]) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(len(index), len(situations), len(outcomes)))


def fit_cells(
    counts: np.ndarray,
    index: SeatIndex,
    situations: list[str],
    outcomes: list[str],
    prefix: str,
    scale: float = 1.0,
    unit: float = 1.0,
    kappa: float | Sequence[float] | None = None,
) -> CellFit:
    width = len(situations) * len(outcomes)
    totals = np.zeros((len(index.who), width))
    for start in range(0, len(index), CHUNK_SEATS):
        stop = min(start + CHUNK_SEATS, len(index))
        block = np.asarray(counts[start:stop], dtype=float).reshape(stop - start, width) / unit
        onehot = sparse.csr_matrix((np.ones(stop - start), (index.who_code[start:stop], np.arange(stop - start))), shape=(len(index.who), stop - start))
        totals += onehot @ block
    totals = totals.reshape(len(index.who), len(situations), len(outcomes))
    worlds = np.zeros((len(situations), len(index.positions), len(outcomes)))
    for code in range(len(index.positions)):
        pooled = totals[index.who_position == code].sum(axis=0)
        worlds[:, code] = (pooled + 1.0) / (pooled.sum(axis=1, keepdims=True) + len(outcomes))
    given = None if kappa is None else (list(kappa) if isinstance(kappa, Sequence) else [kappa] * len(situations))
    kappa = np.array(
        [scale * (given[step] if given is not None else _kappa(totals[:, step], worlds[step, index.who_position])) for step in range(len(situations))]
    )
    report = {
        situation: {
            "rows": int(totals[:, step].sum()),
            "kappa": round(float(kappa[step]), 2),
            "world": {position: [round(float(v), 4) for v in worlds[step, code]] for code, position in enumerate(index.positions)},
        }
        for step, situation in enumerate(situations)
    }
    return CellFit(prefix, list(situations), list(outcomes), totals, worlds, kappa, report, unit)


def cell_values(fit: CellFit, counts: np.ndarray, index: SeatIndex, start: int, stop: int) -> np.ndarray:
    values = np.asarray(counts[start:stop], dtype=float) / fit.unit
    others = fit.totals[index.who_code[start:stop]] - values
    exposure = others.sum(axis=2, keepdims=True)
    world = fit.worlds[:, index.seat_position[start:stop]].transpose(1, 0, 2)
    kappa = fit.kappa[None, :, None]
    return ((world * kappa + others) / (exposure + kappa)).reshape(stop - start, -1)


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
    exposure = fit.totals.sum(axis=2)
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
