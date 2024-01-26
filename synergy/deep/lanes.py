import logging

import numpy as np

from ..config import Settings, get_settings
from ..features.regions import REGIONS, region_of
from ..features.timeline import ParsedTimeline
from ..features.wave import LANE_PREFIX, lane_progress
from ..ingest.store import Store
from ..ingest.window import truncate_timeline

logger = logging.getLogger(__name__)

LANES = ("TOP", "MIDDLE", "BOTTOM")
MAX_MINUTES = 16
OBSERVED = [
    "blue_in_lane",
    "blue_progress",
    "red_in_lane",
    "red_progress",
    "blue_cs_delta",
    "red_cs_delta",
    "laner_gold_diff",
    "blue_jungler_here",
    "red_jungler_here",
    "blue_support_here",
    "red_support_here",
    "blue_ward",
    "red_ward",
    "clock",
]
ANCHORS = ["blue_plate", "red_plate"]
LANE_ROLES = {"TOP": ("TOP",), "MIDDLE": ("MIDDLE",), "BOTTOM": ("BOTTOM", "UTILITY")}


def _in_lane(x: float, y: float, lane: str) -> bool:
    return REGIONS[region_of(x, y, 100)].startswith(LANE_PREFIX[lane])


def encode_lanes(match: dict, timeline: dict) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = min(len(parsed.minutes), MAX_MINUTES)
    by_role: dict[tuple[int, str], int] = {}
    for pid, role in parsed.roles.items():
        if role:
            by_role[(parsed.teams.get(pid, 100), role)] = pid

    plates = {100: {}, 200: {}}
    wards = {100: {}, 200: {}}
    for event in parsed.events:
        minute = int(float(event.get("timestamp", 0)) // 60000)
        if minute >= MAX_MINUTES:
            continue
        actor = int(event.get("killerId", 0) or event.get("creatorId", 0) or 0)
        team = parsed.teams.get(actor)
        if team is None:
            continue
        if event.get("type") == "TURRET_PLATE_DESTROYED":
            position = event.get("position") or {}
            for lane in LANES:
                if _in_lane(float(position.get("x", 0.0)), float(position.get("y", 0.0)), lane):
                    plates[team].setdefault(lane, set()).add(minute)
        elif event.get("type") == "WARD_PLACED":
            role = parsed.roles.get(actor, "")
            for lane, roles in LANE_ROLES.items():
                if role in roles:
                    wards[team].setdefault(lane, set()).add(minute)

    rows = []
    for lane in LANES:
        blue = by_role.get((100, lane))
        red = by_role.get((200, lane))
        if blue is None or red is None:
            continue
        observed = np.zeros((MAX_MINUTES, len(OBSERVED)), dtype=np.float32)
        anchors = np.zeros((MAX_MINUTES, len(ANCHORS)), dtype=np.float32)
        mask = np.zeros(MAX_MINUTES, dtype=bool)
        for minute in range(span):
            filled = False
            for side, pid, offset in ((100, blue, 0), (200, red, 2)):
                positions = parsed.positions.get(pid) or []
                cs = parsed.cs.get(pid) or []
                if minute >= len(positions):
                    continue
                x, y = positions[minute]
                here = _in_lane(x, y, lane)
                observed[minute, offset] = float(here)
                observed[minute, offset + 1] = lane_progress(x, y, 100) if here else 0.0
                observed[minute, 4 + (0 if side == 100 else 1)] = (
                    (cs[minute] - cs[minute - 1]) / 10.0 if 0 < minute < len(cs) else 0.0
                )
                filled = filled or here
            gold_blue = (parsed.gold.get(blue) or [0.0])[min(minute, len(parsed.gold.get(blue) or [0.0]) - 1)]
            gold_red = (parsed.gold.get(red) or [0.0])[min(minute, len(parsed.gold.get(red) or [0.0]) - 1)]
            observed[minute, 6] = (gold_blue - gold_red) / 1000.0
            for slot, (side, role) in enumerate(
                ((100, "JUNGLE"), (200, "JUNGLE"), (100, "UTILITY"), (200, "UTILITY"))
            ):
                other = by_role.get((side, role))
                track = parsed.positions.get(other) or []
                if other is not None and minute < len(track):
                    observed[minute, 7 + slot] = float(_in_lane(track[minute][0], track[minute][1], lane))
            observed[minute, 11] = float(minute in wards[100].get(lane, set()))
            observed[minute, 12] = float(minute in wards[200].get(lane, set()))
            observed[minute, 13] = minute / MAX_MINUTES
            anchors[minute, 0] = float(minute in plates[100].get(lane, set()))
            anchors[minute, 1] = float(minute in plates[200].get(lane, set()))
            mask[minute] = filled
        if mask.sum() >= 6:
            rows.append(
                {
                    "match_id": match["metadata"]["matchId"],
                    "lane": lane,
                    "observed": observed,
                    "anchors": anchors,
                    "mask": mask,
                }
            )
    return rows


def build_lane_sequences(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    rows: list[dict] = []
    with Store(settings) as store:
        for match_id in store.match_ids():
            if store.has_window(match_id):
                window = store.load_window(match_id)
            elif store.has_timeline(match_id):
                window = truncate_timeline(store.load_timeline(match_id))
            else:
                continue
            try:
                rows.extend(encode_lanes(store.load_match(match_id), window))
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("skipped %s: %s", match_id, exc)
    if not rows:
        raise RuntimeError("no lane sequences, run the crawl and window steps first")
    payload = {
        "observed": np.stack([row["observed"] for row in rows]),
        "anchors": np.stack([row["anchors"] for row in rows]),
        "mask": np.stack([row["mask"] for row in rows]),
        "lane": np.array([LANES.index(row["lane"]) for row in rows], dtype=np.int16),
        "match_id": np.array([row["match_id"] for row in rows]),
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(settings.processed_dir / "lane_sequences.npz", **payload)
    return {
        "lanes": len(rows),
        "matches": int(len(set(payload["match_id"]))),
        "minutes": int(payload["mask"].sum()),
        "plate_minutes": int((payload["anchors"].sum(axis=2) > 0).sum()),
    }


def load_lane_sequences(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    with np.load(settings.processed_dir / "lane_sequences.npz", allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
