import logging
import math

import numpy as np

from ..config import Settings, get_settings
from ..features.regions import MAP_SPAN, REGIONS, region_of
from ..features.timeline import VISION_WARDS, ParsedTimeline
from ..ingest.store import Store
from ..ingest.window import truncate_timeline

logger = logging.getLogger(__name__)

MAX_MINUTES = 16
NO_EVENT_REGION = len(REGIONS)
OWN_JUNGLE = {"JUNGLE_OWN_TOPSIDE", "JUNGLE_OWN_BOTSIDE"}
NUMERIC_FEATURES = [
    "gold_delta",
    "xp_delta",
    "cs_delta",
    "jungle_cs_delta",
    "level",
    "travel",
    "damage_done_delta",
    "damage_taken_delta",
    "kills",
    "deaths",
    "assists",
    "wards_placed",
    "wards_killed",
    "plates",
    "objectives",
    "buildings",
    "gold_share",
    "ally_jungler_distance",
    "enemy_jungler_distance",
    "enemy_jungler_invading",
    "lane_opponent_distance",
    "ally_support_distance",
    "enemy_support_distance",
    "clock",
]
COUNTER_BASE = 8
COUNTER_SLOTS = 8
POSITION_EVENTS = {
    "CHAMPION_KILL",
    "CHAMPION_SPECIAL_KILL",
    "TURRET_PLATE_DESTROYED",
    "BUILDING_KILL",
    "ELITE_MONSTER_KILL",
}
ACTOR_KEYS = ("killerId", "creatorId", "participantId")
EVENT_SLOTS = {
    "WARD_PLACED": 3,
    "WARD_KILL": 4,
    "TURRET_PLATE_DESTROYED": 5,
    "ELITE_MONSTER_KILL": 6,
    "BUILDING_KILL": 7,
}


def _minute(event: dict) -> int:
    return int(float(event.get("timestamp", 0)) // 60000)


def _actors(event: dict) -> set[int]:
    ids = {int(event[key]) for key in ACTOR_KEYS if event.get(key)}
    ids.update(int(value) for value in event.get("assistingParticipantIds") or [])
    return ids


def _delta(series: list[float], index: int) -> float:
    if index >= len(series):
        return 0.0
    return series[index] - (series[index - 1] if index > 0 else 0.0)


def encode_match(match: dict, timeline: dict) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    span = min(len(parsed.minutes), MAX_MINUTES)
    champions, wins = {}, {}
    for participant in match["info"]["participants"]:
        pid = int(participant.get("participantId", 0)) or parsed.puuid_to_pid.get(participant["puuid"], 0)
        champions[pid] = participant.get("championName", "")
        wins[int(participant.get("teamId", 0))] = bool(participant.get("win"))
    junglers = {team: parsed.role_pid(team, "JUNGLE") for team in (100, 200)}
    supports = {team: parsed.role_pid(team, "UTILITY") for team in (100, 200)}

    counters = {pid: np.zeros((MAX_MINUTES, COUNTER_SLOTS), dtype=np.float32) for pid in parsed.pid_to_puuid}
    event_regions = {
        pid: np.full(MAX_MINUTES, NO_EVENT_REGION, dtype=np.int16) for pid in parsed.pid_to_puuid
    }
    for event in parsed.events:
        kind = event.get("type")
        minute = _minute(event)
        if minute >= MAX_MINUTES:
            continue
        if kind in ("WARD_PLACED", "WARD_KILL") and event.get("wardType") not in VISION_WARDS:
            continue
        if kind == "CHAMPION_KILL":
            for slot, key in ((0, "killerId"), (1, "victimId")):
                actor = int(event.get(key, 0))
                if actor in counters:
                    counters[actor][minute, slot] += 1
            for helper in event.get("assistingParticipantIds") or []:
                if int(helper) in counters:
                    counters[int(helper)][minute, 2] += 1
        elif kind in EVENT_SLOTS:
            for actor in _actors(event):
                if actor in counters:
                    counters[actor][minute, EVENT_SLOTS[kind]] += 1
        position = event.get("position")
        if kind in POSITION_EVENTS and position:
            for actor in _actors(event):
                if actor in counters and event_regions[actor][minute] == NO_EVENT_REGION:
                    event_regions[actor][minute] = region_of(
                        float(position.get("x", 0.0)),
                        float(position.get("y", 0.0)),
                        parsed.teams.get(actor, 100),
                    )

    team_gold = np.zeros((2, MAX_MINUTES), dtype=np.float32)
    for pid, series in parsed.gold.items():
        side = 0 if parsed.teams.get(pid, 100) == 100 else 1
        for index in range(min(len(series), MAX_MINUTES)):
            team_gold[side, index] += series[index]

    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        team = parsed.teams.get(pid, 100)
        enemy_team = 200 if team == 100 else 100
        positions = parsed.positions.get(pid) or []
        if len(positions) < 8:
            continue
        role = parsed.roles.get(pid, "")
        counterparts = {
            17: junglers.get(team) if junglers.get(team) != pid else None,
            18: junglers.get(enemy_team),
            20: parsed.role_pid(enemy_team, role) if role else None,
            21: supports.get(team) if supports.get(team) != pid else None,
            22: supports.get(enemy_team),
        }
        tracks = {slot: (parsed.positions.get(other) or []) for slot, other in counterparts.items() if other}
        enemy_track = parsed.positions.get(junglers.get(enemy_team)) or []
        gold = parsed.gold.get(pid) or []
        cs = parsed.cs.get(pid) or []
        jungle = parsed.jungle_cs.get(pid) or []
        levels = parsed.levels.get(pid) or []
        regions = np.full(MAX_MINUTES, REGIONS.index("UNKNOWN"), dtype=np.int16)
        numeric = np.zeros((MAX_MINUTES, len(NUMERIC_FEATURES)), dtype=np.float32)
        mask = np.zeros(MAX_MINUTES, dtype=bool)
        for index in range(min(span, len(positions))):
            x, y = positions[index]
            regions[index] = region_of(x, y, team)
            mask[index] = True
            numeric[index, 0] = _delta(gold, index) / 1000.0
            numeric[index, 1] = _delta(parsed.xp.get(pid) or [], index) / 1000.0
            numeric[index, 2] = (_delta(cs, index) - _delta(jungle, index)) / 10.0
            numeric[index, 3] = _delta(jungle, index) / 10.0
            numeric[index, 4] = (levels[index] if index < len(levels) else 1.0) / 18.0
            if index > 0:
                numeric[index, 5] = (
                    math.hypot(x - positions[index - 1][0], y - positions[index - 1][1]) / MAP_SPAN
                )
            numeric[index, 6] = _delta(parsed.damage_done.get(pid) or [], index) / 1000.0
            numeric[index, 7] = _delta(parsed.damage_taken.get(pid) or [], index) / 1000.0
            numeric[index, COUNTER_BASE : COUNTER_BASE + COUNTER_SLOTS] = counters[pid][index]
            side = 0 if team == 100 else 1
            total = team_gold[side, index]
            numeric[index, 16] = (gold[index] / total) if total > 0 and index < len(gold) else 0.0
            for slot, track in tracks.items():
                if index < len(track):
                    numeric[index, slot] = (
                        math.hypot(x - track[index][0], y - track[index][1]) / MAP_SPAN
                    )
            if index < len(enemy_track):
                ex, ey = enemy_track[index]
                numeric[index, 19] = float(REGIONS[region_of(ex, ey, team)] in OWN_JUNGLE)
            numeric[index, 23] = index / MAX_MINUTES
        rows.append(
            {
                "match_id": match["metadata"]["matchId"],
                "puuid": puuid,
                "champion": champions.get(pid, ""),
                "position": parsed.roles.get(pid, "") or "UNKNOWN",
                "team_id": team,
                "win": int(bool(wins.get(team, False))),
                "regions": regions,
                "event_regions": event_regions[pid],
                "numeric": numeric,
                "mask": mask,
            }
        )
    return rows


def _window(store: Store, match_id: str) -> dict | None:
    if store.has_window(match_id):
        return store.load_window(match_id)
    if store.has_timeline(match_id):
        return truncate_timeline(store.load_timeline(match_id))
    return None


def build_sequences(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    rows: list[dict] = []
    with Store(settings) as store:
        match_ids = store.match_ids()
        for match_id in match_ids:
            window = _window(store, match_id)
            if window is None:
                continue
            try:
                rows.extend(encode_match(store.load_match(match_id), window))
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("skipped %s: %s", match_id, exc)
    if not rows:
        raise RuntimeError("no timelines available, run the crawl and window steps first")

    champions = sorted({row["champion"] for row in rows})
    positions = sorted({row["position"] for row in rows})
    champion_index = {name: index for index, name in enumerate(champions)}
    position_index = {name: index for index, name in enumerate(positions)}

    payload = {
        "regions": np.stack([row["regions"] for row in rows]),
        "event_regions": np.stack([row["event_regions"] for row in rows]),
        "numeric": np.stack([row["numeric"] for row in rows]),
        "mask": np.stack([row["mask"] for row in rows]),
        "champion": np.array([champion_index[row["champion"]] for row in rows], dtype=np.int16),
        "position": np.array([position_index[row["position"]] for row in rows], dtype=np.int16),
        "side": np.array([0 if row["team_id"] == 100 else 1 for row in rows], dtype=np.int16),
        "win": np.array([row["win"] for row in rows], dtype=np.int8),
        "puuid": np.array([row["puuid"] for row in rows]),
        "match_id": np.array([row["match_id"] for row in rows]),
        "champion_names": np.array(champions),
        "position_names": np.array(positions),
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(settings.processed_dir / "sequences.npz", **payload)
    logger.info("wrote %s sequences", len(rows))
    return {
        "sequences": len(rows),
        "matches": int(len(set(payload["match_id"]))),
        "players": int(len(set(payload["puuid"]))),
        "champions": len(champions),
        "minutes": int(payload["mask"].sum()),
    }


def load_sequences(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    with np.load(settings.processed_dir / "sequences.npz", allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
