import logging

from ..config import Settings, get_settings
from ..ingest.store import Store

logger = logging.getLogger(__name__)


def load_from_archive(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    report = {"matches": 0, "timelines": 0, "windows": 0, "skipped": 0}
    with Store(settings) as store:
        paths = sorted(settings.match_dir.glob("*.json.gz"))
        for index, path in enumerate(paths):
            match_id = path.name.removesuffix(".json.gz")
            try:
                store.save_match(store.load_match(match_id))
            except (OSError, KeyError, ValueError) as exc:
                logger.warning("skipped %s: %s", match_id, exc)
                report["skipped"] += 1
                continue
            report["matches"] += 1
            if store.has_timeline(match_id):
                store.ingest_timeline(match_id, store.load_timeline(match_id))
                report["timelines"] += 1
            if store.has_window(match_id):
                store.save_window(match_id, store.load_window(match_id))
                report["windows"] += 1
            if index % 200 == 0:
                logger.info("loaded %s of %s matches", index, len(paths))
        report["counts"] = store.counts()
    return report
