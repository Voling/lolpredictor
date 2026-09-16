import json

import numpy as np
from psycopg.rows import tuple_row

from ..config import Settings, get_settings
from ..features.anchors import plausible
from ..features.posterior import (
    SIGMA_FILE,
    TOP,
    anchor_points,
    bridge_at,
    dead_mask,
    death_spans,
    fit_sigma,
    known_points,
    load_sigma,
    nearer_residuals,
    region_mass,
    top_regions,
    wave_mass,
)
from ..features.regions import REGIONS, regions_of
from ..ingest.store import Store

SPAN = 16
SEATS = 10
MAX_EVENTS = 1024
LEAD_SCALE = 2.33
PAD = 0
SIGMA_SAMPLE = 3000

KINDS = [
    "PAD",
    "ITEM_PURCHASED",
    "ITEM_DESTROYED",
    "ITEM_UNDO",
    "ITEM_SOLD",
    "SKILL_LEVEL_UP",
    "LEVEL_UP",
    "WARD_PLACED",
    "WARD_KILL",
    "CHAMPION_KILL",
    "CHAMPION_SPECIAL_KILL",
    "TURRET_PLATE_DESTROYED",
    "BUILDING_KILL",
    "ELITE_MONSTER_KILL",
    "DRAGON_SOUL_GIVEN",
    "GAME_END",
    "FEAT_UPDATE",
    "OBJECTIVE_BOUNTY_PRESTART",
    "OBJECTIVE_BOUNTY_FINISH",
    "CHAMPION_TRANSFORM",
]
KIND_INDEX = {name: index for index, name in enumerate(KINDS)}
OTHER = len(KINDS)

FRAME_QUERY = (
    "SELECT f.match_id, f.puuid, f.minute, f.x, f.y, f.total_gold, f.minions, f.jungle_minions,"
    " f.level FROM frames f WHERE f.minute < %s AND f.match_id = ANY(%s)"
)
EVENT_QUERY = (
    "SELECT e.match_id, e.timestamp_ms, e.type, e.actor, e.victim, e.assists, e.x, e.y"
    " FROM events e WHERE e.minute < %s AND e.match_id = ANY(%s)"
    " ORDER BY e.match_id, e.timestamp_ms, e.event_index"
)
SEAT_QUERY = (
    "SELECT match_id, puuid, team_id, position, champion_name, win"
    " FROM participations WHERE match_id = ANY(%s) ORDER BY match_id, team_id, position"
)


def seat_table(settings: Settings, match_ids: list[str]) -> dict:
    store = Store(settings)
    try:
        with store.conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(SEAT_QUERY, (match_ids,))
            rows = cursor.fetchall()
    finally:
        store.close()
    seats: dict[str, dict] = {}
    for match_id, puuid, team, position, champion, win in rows:
        entry = seats.setdefault(match_id, {"puuid": [], "team": [], "position": [], "champion": [], "win": None})
        entry["puuid"].append(puuid)
        entry["team"].append(int(team or 100))
        entry["position"].append(position or "")
        entry["champion"].append(champion or "")
        if int(team or 100) == 100:
            entry["win"] = int(bool(win))
    return {key: value for key, value in seats.items() if len(value["puuid"]) == SEATS}


def frame_tracks(settings: Settings, match_ids: list[str], seats: dict) -> dict:
    store = Store(settings)
    try:
        with store.conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(FRAME_QUERY, (SPAN, match_ids))
            rows = cursor.fetchall()
    finally:
        store.close()
    tracks: dict[str, dict] = {}
    for match_id, puuid, minute, x, y, gold, minions, jungle, level in rows:
        entry = seats.get(match_id)
        if entry is None:
            continue
        spot = tracks.setdefault(
            match_id,
            {
                "xy": np.zeros((SEATS, SPAN, 2)),
                "gold": np.zeros((SEATS, SPAN)),
                "cs": np.zeros((SEATS, SPAN)),
                "level": np.ones((SEATS, SPAN), np.int16),
                "known": np.zeros((SEATS, SPAN), bool),
            },
        )
        try:
            seat = entry["puuid"].index(puuid)
        except ValueError:
            continue
        if not 0 <= minute < SPAN:
            continue
        spot["xy"][seat, minute] = (float(x or 0.0), float(y or 0.0))
        spot["gold"][seat, minute] = float(gold or 0.0)
        spot["cs"][seat, minute] = float(minions or 0.0) - float(jungle or 0.0)
        spot["level"][seat, minute] = int(level or 1)
        spot["known"][seat, minute] = x is not None
    return tracks


def event_rows(settings: Settings, match_ids: list[str]) -> list[tuple]:
    store = Store(settings)
    try:
        with store.conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(EVENT_QUERY, (SPAN, match_ids))
            return cursor.fetchall()
    finally:
        store.close()


def _by_match(rows: list[tuple]) -> dict[str, list[tuple]]:
    grouped: dict[str, list[tuple]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row[1:])
    return grouped


def _seat_points(match: dict, spot: dict, events: list[tuple]) -> tuple[list, list]:
    seat_of = {puuid: index + 1 for index, puuid in enumerate(match["puuid"])}
    blue_of = [1 if team == 100 else 0 for team in match["team"]]
    levelled = []
    for stamp, kind, actor, victim, assists, x, y in events:
        seat = seat_of.get(victim, 0)
        when = min(max(int(float(stamp or 0) // 60000.0), 0), SPAN - 1)
        level = int(spot["level"][seat - 1, when]) if seat else 1
        levelled.append((stamp, kind, actor, victim, assists, x, y, level))
    certain, claimed, kills = anchor_points(levelled, seat_of, blue_of)
    points, spans = [], []
    for seat in range(SEATS):
        team = 100 if blue_of[seat] else 200
        span = death_spans(kills.get(seat + 1, []))
        spans.append(span)
        points.append(
            known_points(
                spot["xy"][seat], spot["known"][seat], span,
                certain.get(seat + 1, []), claimed.get(seat + 1, []), team,
            )
        )
    return points, spans


def calibrate_sigma(settings: Settings, match_ids: list[str]) -> dict:
    seats = seat_table(settings, match_ids)
    wanted = sorted(seats)
    tracks = frame_tracks(settings, wanted, seats)
    grouped = _by_match(event_rows(settings, wanted))
    residuals: dict[str, list[tuple[float, float]]] = {}
    for match_id in wanted:
        spot, match = tracks.get(match_id), seats[match_id]
        if spot is None:
            continue
        seat_of = {puuid: index + 1 for index, puuid in enumerate(match["puuid"])}
        blue_of = [1 if team == 100 else 0 for team in match["team"]]
        levelled = [(s, k, a, v, ass, x, y, 1) for s, k, a, v, ass, x, y in grouped.get(match_id, [])]
        _, claimed, kills = anchor_points(levelled, seat_of, blue_of)
        for seat in range(SEATS):
            frames_only = known_points(
                spot["xy"][seat], spot["known"][seat], death_spans(kills.get(seat + 1, [])), [], [],
                100 if blue_of[seat] else 200,
            )
            frames_only = frames_only[np.isclose(frames_only[:, 0], np.round(frames_only[:, 0]))]
            witnessed = [
                (m, x, y) for m, x, y in claimed.get(seat + 1, [])
                if plausible([tuple(p) for p in frames_only], m, (x, y))
            ]
            if not witnessed:
                continue
            role = match["position"][seat] or "UNKNOWN"
            residuals.setdefault(role, []).extend(nearer_residuals(frames_only, witnessed))
    sigma = fit_sigma(residuals)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    report = {"sigma": sigma, "anchors": {role: len(rows) for role, rows in residuals.items()}, "matches": len(wanted)}
    with open(settings.model_dir / SIGMA_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def match_stream(settings: Settings, match_ids: list[str]) -> dict:
    seats = seat_table(settings, match_ids)
    wanted = sorted(seats)
    tracks = frame_tracks(settings, wanted, seats)
    grouped = _by_match(event_rows(settings, wanted))
    sigma = load_sigma(settings)

    count = len(wanted)
    kind = np.zeros((count, MAX_EVENTS), np.int16)
    actor = np.zeros((count, MAX_EVENTS), np.int16)
    victim = np.zeros((count, MAX_EVENTS), np.int16)
    region = np.zeros((count, MAX_EVENTS), np.int16)
    clock = np.zeros((count, MAX_EVENTS), np.float32)
    seat_region_top = np.zeros((count, MAX_EVENTS, TOP), np.int8)
    seat_region_p = np.zeros((count, MAX_EVENTS, TOP), np.float16)
    wave = np.zeros((count, MAX_EVENTS, 3), np.float16)
    lead = np.zeros((count, MAX_EVENTS), np.float32)
    mask = np.zeros((count, MAX_EVENTS), bool)
    filled = np.zeros(count, np.int32)

    for row, match_id in enumerate(wanted):
        match, spot = seats[match_id], tracks.get(match_id)
        events = grouped.get(match_id, [])[:MAX_EVENTS]
        if not events:
            continue
        seat_of = {puuid: index + 1 for index, puuid in enumerate(match["puuid"])}
        blue_of = [1 if team == 100 else 0 for team in match["team"]]
        blue = np.array(blue_of, dtype=bool)
        stamps = np.array([float(stamp or 0) for stamp, *_ in events])
        minutes = stamps / 60000.0
        floored = np.clip(minutes.astype(int), 0, SPAN - 1)
        length = len(events)
        for at, (stamp, kind_name, who, hurt, _, x, y) in enumerate(events):
            kind[row, at] = KIND_INDEX.get(kind_name, OTHER)
            actor[row, at] = seat_of.get(who, 0)
            victim[row, at] = seat_of.get(hurt, 0)
            seat = int(actor[row, at])
            acting = blue_of[seat - 1] if seat else 1
            region[row, at] = (
                regions_of(np.array([x or 0.0]), np.array([y or 0.0]), np.array([100 if acting else 200]))[0] + 1
                if x is not None
                else 0
            )
        clock[row, :length] = minutes / SPAN
        mask[row, :length] = True
        filled[row] = length
        if spot is None:
            continue
        gold = spot["gold"][:, floored]
        lead[row, :length] = (gold[blue].sum(axis=0) - gold[~blue].sum(axis=0)) / 1000.0 / LEAD_SCALE

        points, spans = _seat_points(match, spot, events)
        actors = actor[row, :length]
        for seat in range(SEATS):
            tokens = np.flatnonzero(actors == seat + 1)
            if len(tokens) == 0:
                continue
            team = 100 if blue_of[seat] else 200
            role = match["position"][seat]
            when = minutes[tokens]
            dead = dead_mask(spans[seat], when)
            mix = bridge_at(points[seat], when, sigma.get(role, sigma["TOP"]))
            masses = region_mass(mix, team)
            masses[dead] = 0.0
            cs = spot["cs"][seat]
            floor = floored[tokens]
            farmed = np.where(floor > 0, cs[floor] - cs[np.clip(floor - 1, 0, None)], 0.0)
            waves = wave_mass(masses, mix, team, role, farmed, dead | (masses.sum(axis=1) == 0))
            order, picked = top_regions(masses)
            live = masses.sum(axis=1) > 0
            seat_region_top[row, tokens[live]] = order[live] + 1
            seat_region_p[row, tokens[live]] = picked[live]
            wave[row, tokens] = waves

    played = filled > 0
    if not played.all():
        wanted = [match_id for match_id, keep in zip(wanted, played) if keep]
        kind, actor, victim, region = kind[played], actor[played], victim[played], region[played]
        clock, mask, lead = clock[played], mask[played], lead[played]
        seat_region_top, seat_region_p, wave = seat_region_top[played], seat_region_p[played], wave[played]
        filled = filled[played]
        count = len(wanted)

    champions = sorted({name for value in seats.values() for name in value["champion"]})
    champion_index = {name: index for index, name in enumerate(champions)}
    roles = ["", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    role_index = {name: index for index, name in enumerate(roles)}

    seat_champion = np.zeros((count, SEATS), np.int16)
    seat_role = np.zeros((count, SEATS), np.int16)
    seat_side = np.zeros((count, SEATS), np.int8)
    seat_puuid = np.empty((count, SEATS), object)
    win = np.zeros(count, np.int8)
    for row, match_id in enumerate(wanted):
        value = seats[match_id]
        for index in range(SEATS):
            seat_champion[row, index] = champion_index.get(value["champion"][index], 0)
            seat_role[row, index] = role_index.get(value["position"][index], 0)
            seat_side[row, index] = 1 if value["team"][index] == 100 else 0
            seat_puuid[row, index] = value["puuid"][index]
        if value["win"] is None:
            raise ValueError(f"{match_id} has no blue side result")
        win[row] = value["win"]
    return {
        "match_id": np.array(wanted),
        "seat_puuid": seat_puuid.astype(str),
        "kind": kind,
        "actor": actor,
        "victim": victim,
        "region": region,
        "seat_region_top": seat_region_top,
        "seat_region_p": seat_region_p,
        "wave": wave,
        "lead": lead,
        "clock": clock,
        "mask": mask,
        "seat_champion": seat_champion,
        "seat_role": seat_role,
        "seat_side": seat_side,
        "win": win,
        "events": int(filled.sum()),
        "truncated": int((filled >= MAX_EVENTS).sum()),
        "dropped_empty": int((~played).sum()),
        "champions": np.array(champions),
        "kinds": np.array(KINDS + ["OTHER"]),
        "regions": np.array(["NONE", *REGIONS]),
        "sigma": np.array(json.dumps(sigma)),
    }


def build_stream(
    settings: Settings | None = None, limit: int | None = None, path: str = "stream.npz"
) -> dict:
    settings = settings or get_settings()
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute("SELECT match_id FROM matches ORDER BY match_id")
            match_ids = [row["match_id"] for row in cursor.fetchall()]
    finally:
        store.close()
    if limit:
        match_ids = match_ids[:limit]
    calibration = None
    if not (settings.model_dir / SIGMA_FILE).exists():
        calibration = calibrate_sigma(settings, match_ids[:SIGMA_SAMPLE])
    built = match_stream(settings, match_ids)
    np.savez_compressed(settings.processed_dir / path, **built)
    return {
        "matches": int(len(built["match_id"])),
        "events": built["events"],
        "truncated_matches": built["truncated"],
        "cap": MAX_EVENTS,
        "champions": int(len(built["champions"])),
        "sigma": json.loads(str(built["sigma"])),
        "calibrated": calibration,
    }
