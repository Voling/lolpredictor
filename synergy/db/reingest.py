import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from ..config import Settings, get_settings
from ..ingest.store import Store

logger = logging.getLogger(__name__)


def _chunk(settings: Settings, match_ids: list[str]) -> tuple[int, int, int]:
    frames = events = skipped = 0
    with Store(settings) as store:
        for match_id in match_ids:
            if not store.has_timeline(match_id):
                skipped += 1
                continue
            try:
                added_frames, added_events = store.ingest_timeline(
                    match_id, store.load_timeline(match_id)
                )
            except (FileNotFoundError, KeyError, OSError) as exc:
                logger.warning("skipped %s: %s", match_id, exc)
                skipped += 1
                continue
            frames += added_frames
            events += added_events
    return frames, events, skipped


def _split(items: list[str], parts: int) -> list[list[str]]:
    if not items:
        return [[]]
    parts = max(1, min(parts, len(items)))
    size = (len(items) + parts - 1) // parts
    return [items[index : index + size] for index in range(0, len(items), size)]


def reingest(
    settings: Settings | None = None, workers: int | None = None, missing: bool = False
) -> dict:
    settings = settings or get_settings()
    with Store(settings) as store:
        match_ids = store.matches_without_frames() if missing else store.match_ids()
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) // 2))
    chunks = _split(match_ids, workers)
    if len(chunks) == 1:
        results = [_chunk(settings, chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            results = list(pool.map(_chunk, repeat(settings), chunks))
    return {
        "matches": len(match_ids),
        "missing_only": missing,
        "frames": sum(item[0] for item in results),
        "events": sum(item[1] for item in results),
        "skipped": sum(item[2] for item in results),
        "workers": len(chunks),
    }
