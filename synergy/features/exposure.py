import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .cells import CHUNK_SEATS, SeatIndex, buffer, dense_counts
from .positions import KEY

EXPOSURE_TABLE = "exposure.parquet"


def seat_exposure(counts, index: SeatIndex, situations: list[str], unit: float = 1.0) -> np.ndarray:
    totals = np.zeros((len(index.who), len(situations)))
    for start in range(0, len(index), CHUNK_SEATS):
        stop = min(start + CHUNK_SEATS, len(index))
        block = np.asarray(counts[start:stop], dtype=float).reshape(stop - start, len(situations), -1).sum(axis=2) / unit
        np.add.at(totals, index.who_code[start:stop], block)
    return totals


def exposure_frame(index: SeatIndex, parts: list[tuple[str, list[str], np.ndarray]]) -> pd.DataFrame:
    frame = pd.DataFrame({"puuid": index.who.get_level_values("puuid"), "position": index.who.get_level_values("position")})
    for prefix, situations, totals in parts:
        for situation, values in zip(situations, totals.T):
            frame[f"{prefix}_{situation}"] = values.astype(np.float32)
    return frame


def write_exposure(settings: Settings | None = None) -> dict:
    from ..ingest.store import Store
    from .fits import TENDENCY_FIT, load_tendencies, reaction_sources, tendency_totals
    from .priority import COUNTS, SITUATIONS, TICK_UNIT, priority_context

    settings = settings or get_settings()
    processed = settings.processed_dir
    seats = pd.read_parquet(processed / "participations.parquet", columns=["match_id", "puuid", "position"])
    index = SeatIndex(seats)
    parts = []
    for number, (prefix, load, situations, outcomes) in enumerate(reaction_sources(settings)):
        path = processed / "buffers" / f"exposure.{number}.npy"
        counts = buffer(path, index, situations, outcomes)
        for frame in load():
            dense_counts(frame, index, situations, outcomes, out=counts)
        parts.append((prefix, situations, seat_exposure(counts, index, situations)))
        del counts
        path.unlink(missing_ok=True)
    table = exposure_frame(index, parts)
    counts_path = processed / "buffers" / COUNTS
    if counts_path.exists():
        store = Store(settings)
        try:
            corpus = set(store.corpus_ids())
        finally:
            store.close()
        narrow = SeatIndex(seats[seats.match_id.isin(corpus)])
        counts = np.load(counts_path, mmap_mode="r")
        if counts.shape[0] == len(narrow):
            lane = exposure_frame(narrow, [("prio", SITUATIONS, seat_exposure(counts, narrow, SITUATIONS, TICK_UNIT))])
            table = table.merge(lane, on=KEY, how="left")
    fits = load_tendencies(settings.model_dir / TENDENCY_FIT)
    context = priority_context(settings)
    totals = tendency_totals(fits, lambda kind: pd.read_parquet(processed / "opportunities.parquet", filters=[("kind", "==", kind)]), context, seats)
    table = table.merge(totals, on=KEY, how="left").fillna(0.0)
    table.to_parquet(processed / EXPOSURE_TABLE, index=False)
    return {"rows": int(len(table)), "columns": int(len(table.columns) - len(KEY)), "path": str(processed / EXPOSURE_TABLE)}
