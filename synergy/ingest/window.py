import logging

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


def build_windows(settings: Settings | None = None, minutes: int = WINDOW_MINUTES) -> dict:
    settings = settings or get_settings()
    written = skipped = 0
    with Store(settings) as store:
        for match_id in store.match_ids():
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
    logger.info("wrote %s truncated timelines", written)
    return {"windows": written, "skipped": skipped, "minutes": minutes}
