import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from ..config import Settings, get_settings
from .store import Store

logger = logging.getLogger(__name__)

WINDOW_MINUTES = 15


def truncate_timeline(timeline: dict, minutes: int = WINDOW_MINUTES) -> dict:
    limit = minutes * 60000
    info = timeline.get("info", {})
    frames = []
    for frame in info.get("frames") or []:
        if int(frame.get("timestamp", 0)) > limit:
            continue
        frames.append(
            {
                "timestamp": frame.get("timestamp"),
                "participantFrames": frame.get("participantFrames") or {},
                "events": [
                    event
                    for event in frame.get("events") or []
                    if int(event.get("timestamp", 0)) <= limit
                ],
            }
        )
    return {
        "metadata": timeline.get("metadata", {}),
        "info": {
            "frameInterval": info.get("frameInterval"),
            "participants": info.get("participants") or [],
            "frames": frames,
            "windowMinutes": minutes,
        },
    }


def _window_chunk(settings: Settings, match_ids: list[str], minutes: int) -> tuple[int, int]:
    written = skipped = 0
    with Store(settings) as store:
        for match_id in match_ids:
            if not store.has_timeline(match_id):
                skipped += 1
                continue
            try:
                truncated = truncate_timeline(store.load_timeline(match_id), minutes)
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("skipped %s: %s", match_id, exc)
                skipped += 1
                continue
            if len(truncated["info"]["frames"]) < 8:
                skipped += 1
                continue
            store.save_window(match_id, truncated)
            written += 1
    return written, skipped


def build_windows(
    settings: Settings | None = None, minutes: int = WINDOW_MINUTES, workers: int | None = None
) -> dict:
    settings = settings or get_settings()
    with Store(settings) as store:
        match_ids = store.match_ids()
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) // 2))
    chunks = _split(match_ids, workers)
    if len(chunks) == 1:
        results = [_window_chunk(settings, chunks[0], minutes)]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            results = list(pool.map(_window_chunk, repeat(settings), chunks, repeat(minutes)))
    written = sum(item[0] for item in results)
    skipped = sum(item[1] for item in results)
    logger.info("wrote %s truncated timelines across %s workers", written, len(chunks))
    return {"windows": written, "skipped": skipped, "minutes": minutes}


def _split(items: list[str], parts: int) -> list[list[str]]:
    if not items:
        return [[]]
    parts = max(1, min(parts, len(items)))
    size = (len(items) + parts - 1) // parts
    return [items[index : index + size] for index in range(0, len(items), size)]
