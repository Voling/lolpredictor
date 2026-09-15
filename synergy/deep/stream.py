import numpy as np
from psycopg.rows import tuple_row

from ..config import Settings, get_settings
from ..features.regions import REGIONS, regions_of
from ..ingest.store import Store

SPAN = 16
SEATS = 10
MAX_EVENTS = 1024
LEAD_SCALE = 2.33
PAD = 0

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
    "SELECT f.match_id, f.puuid, f.minute, f.x, f.y, f.total_gold"
    " FROM frames f WHERE f.minute < %s AND f.match_id = ANY(%s)"
)

EVENT_QUERY = (
    "SELECT e.match_id, e.timestamp_ms, e.type, e.actor, e.victim, e.x, e.y"
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
    for match_id, puuid, minute, x, y, gold in rows:
        entry = seats.get(match_id)
        if entry is None:
            continue
        spot = tracks.setdefault(
            match_id,
            {
                "x": np.zeros((SEATS, SPAN)),
                "y": np.zeros((SEATS, SPAN)),
                "gold": np.zeros((SEATS, SPAN)),
                "known": np.zeros((SEATS, SPAN), bool),
            },
        )
        try:
            seat = entry["puuid"].index(puuid)
        except ValueError:
            continue
        if not 0 <= minute < SPAN:
            continue
        spot["x"][seat, minute] = float(x or 0.0)
        spot["y"][seat, minute] = float(y or 0.0)
        spot["gold"][seat, minute] = float(gold or 0.0)
        spot["known"][seat, minute] = x is not None
    return tracks


def match_stream(settings: Settings, match_ids: list[str]) -> dict:
    seats = seat_table(settings, match_ids)
    wanted = sorted(seats)
    tracks = frame_tracks(settings, wanted, seats)
    store = Store(settings)
    try:
        with store.conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(EVENT_QUERY, (SPAN, wanted))
            rows = cursor.fetchall()
    finally:
        store.close()

    order = {match_id: index for index, match_id in enumerate(wanted)}
    count = len(wanted)
    kind = np.zeros((count, MAX_EVENTS), np.int16)
    actor = np.zeros((count, MAX_EVENTS), np.int16)
    victim = np.zeros((count, MAX_EVENTS), np.int16)
    region = np.zeros((count, MAX_EVENTS), np.int16)
    clock = np.zeros((count, MAX_EVENTS), np.float32)
    seat_region = np.zeros((count, MAX_EVENTS), np.int16)
    lead = np.zeros((count, MAX_EVENTS), np.float32)
    mask = np.zeros((count, MAX_EVENTS), bool)

    filled = np.zeros(count, np.int32)
    seat_of = {
        match_id: {puuid: index + 1 for index, puuid in enumerate(value["puuid"])}
        for match_id, value in seats.items()
    }
    blue_of = {
        match_id: [1 if team == 100 else 0 for team in value["team"]]
        for match_id, value in seats.items()
    }
    for match_id, stamp, kind_name, who, hurt, x, y in rows:
        row = order.get(match_id)
        if row is None:
            continue
        at = filled[row]
        if at >= MAX_EVENTS:
            continue
        lookup = seat_of[match_id]
        kind[row, at] = KIND_INDEX.get(kind_name, OTHER)
        actor[row, at] = lookup.get(who, 0)
        victim[row, at] = lookup.get(hurt, 0)
        seat = int(actor[row, at])
        acting = blue_of[match_id][seat - 1] if seat else 1
        region[row, at] = (
            regions_of(
                np.array([x or 0.0]), np.array([y or 0.0]), np.array([100 if acting else 200])
            )[0]
            + 1
            if x is not None
            else 0
        )
        clock[row, at] = float(stamp or 0) / 60000.0 / SPAN
        spot = tracks.get(match_id)
        when = min(max(int(float(stamp or 0) // 60000.0), 0), SPAN - 1)
        if spot is not None:
            blue = np.array(blue_of[match_id], dtype=bool)
            gold = spot["gold"][:, when]
            lead[row, at] = float(gold[blue].sum() - gold[~blue].sum()) / 1000.0 / LEAD_SCALE
            if seat and spot["known"][seat - 1, when]:
                side = 100 if blue_of[match_id][seat - 1] else 200
                seat_region[row, at] = (
                    regions_of(
                        np.array([spot["x"][seat - 1, when]]),
                        np.array([spot["y"][seat - 1, when]]),
                        np.array([side]),
                    )[0]
                    + 1
                )
        mask[row, at] = True
        filled[row] = at + 1

    played = filled > 0
    if not played.all():
        wanted = [match_id for match_id, keep in zip(wanted, played) if keep]
        kind, actor, victim = kind[played], actor[played], victim[played]
        region, clock, mask = region[played], clock[played], mask[played]
        seat_region, lead = seat_region[played], lead[played]
        filled = filled[played]
        order = {match_id: index for index, match_id in enumerate(wanted)}
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
    for match_id, value in seats.items():
        row = order.get(match_id)
        if row is None:
            continue
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
        "seat_region": seat_region,
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
    built = match_stream(settings, match_ids)
    np.savez_compressed(settings.processed_dir / path, **built)
    return {
        "matches": int(len(built["match_id"])),
        "events": built["events"],
        "truncated_matches": built["truncated"],
        "cap": MAX_EVENTS,
        "champions": int(len(built["champions"])),
    }
