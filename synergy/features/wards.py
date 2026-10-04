import numpy as np
import pandas as pd

from ..deep.stream import _seat_points, parsed_inputs
from .posterior import bridge_at, load_sigma, radius_mass, region_mass
from .reaction import WARD_ZONES, ZONE_COLUMNS
from .regions import REGIONS, region_of
from .timeline import VISION_WARDS, ParsedTimeline
from .wave import LANE_PREFIX, wave_states

FRAME_MS = 60000.0
GANK_RADIUS = 2200.0
INVADE_LEAD = 90000.0
ENEMY_JUNGLE = {"JUNGLE_ENEMY_TOPSIDE", "JUNGLE_ENEMY_BOTSIDE"}


def _ward_zone(zone: pd.Series) -> np.ndarray:
    z = zone.fillna("").astype(str)
    return np.select(
        [z.str.endswith("_OWN") & z.str.startswith("LANE"), z.str.endswith("_NEUTRAL"), z.str.endswith("_ENEMY") & z.str.startswith("LANE"),
         z.str.startswith("JUNGLE_OWN"), z.str.startswith("JUNGLE_ENEMY"), z.str.startswith("RIVER")],
        ["lane_own_side", "lane_middle", "lane_enemy_side", "own_jungle", "enemy_jungle", "river"],
        default="other",
    )


ZONE_OF_REGION = np.eye(len(WARD_ZONES), dtype=np.float32)[pd.Index(WARD_ZONES).get_indexer(_ward_zone(pd.Series(REGIONS)))]


def _jungler_events(parsed: ParsedTimeline, jungler: int | None, span: int) -> tuple[list[float], list[dict]]:
    invades: list[float] = []
    ganks: list[dict] = []
    if jungler is None:
        return invades, ganks
    team = parsed.teams.get(jungler, 100)
    track = parsed.positions.get(jungler) or []
    previous = None
    for minute in range(min(len(track), span + 1)):
        zone = REGIONS[region_of(track[minute][0], track[minute][1], team)]
        if zone in ENEMY_JUNGLE and previous not in ENEMY_JUNGLE:
            invades.append(minute * FRAME_MS)
        previous = zone
    for event in parsed.kill_events():
        if jungler not in parsed.involved(event):
            continue
        position = event.get("position") or {}
        if not position:
            continue
        ganks.append(
            {
                "timestamp": float(event.get("timestamp", 0)),
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
            }
        )
    return invades, ganks


def ward_posterior(parsed: ParsedTimeline, placed: list[tuple[str, float]], sigma: dict[str, list[float]]) -> tuple[np.ndarray, dict]:
    match, spot, events = parsed_inputs(parsed)
    points, _ = _seat_points(match, spot, events)
    seat_of = {puuid: seat for seat, puuid in enumerate(match["puuid"])}
    by_seat: dict[int, list[int]] = {}
    for row, (puuid, _) in enumerate(placed):
        by_seat.setdefault(seat_of[puuid], []).append(row)
    count = len(placed)
    masses = np.zeros((count, len(WARD_ZONES)))
    mix = {"p0": np.zeros((count, 2)), "p1": np.zeros((count, 2)), "w": np.zeros(count), "s0": np.full(count, np.inf), "s1": np.full(count, np.inf)}
    for seat, rows in by_seat.items():
        minutes = np.array([placed[row][1] for row in rows]) / FRAME_MS
        found = bridge_at(points[seat], minutes, sigma.get(match["position"][seat], sigma["TOP"]))
        masses[rows] = region_mass(found, match["team"][seat]) @ ZONE_OF_REGION
        for key, values in mix.items():
            values[rows] = found[key]
    return masses, mix


def _next_gank(ganks: list[dict], timestamp: float) -> tuple[float, float]:
    upcoming = [gank for gank in ganks if gank["timestamp"] >= timestamp]
    if not upcoming:
        return np.nan, np.nan
    first = min(upcoming, key=lambda gank: gank["timestamp"])
    return first["x"], first["y"]


def ward_rows(
    match: dict, timeline: dict, parsed: ParsedTimeline | None = None, sigma: dict[str, list[float]] | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    sigma = sigma or load_sigma()
    span = len(parsed.minutes) - 1
    lane_states = {
        pid: wave_states(parsed, pid, span)
        for pid in parsed.pid_to_puuid
        if parsed.roles.get(pid) in LANE_PREFIX
    }
    junglers = {team: parsed.role_pid(team, "JUNGLE") for team in (100, 200)}
    jungler_moves = {team: _jungler_events(parsed, junglers[team], span) for team in (100, 200)}

    placed = []
    for event in parsed.events:
        if event.get("type") != "WARD_PLACED" or event.get("wardType") not in VISION_WARDS:
            continue
        pid = int(event.get("creatorId", 0))
        puuid = parsed.pid_to_puuid.get(pid)
        if puuid is None:
            continue
        timestamp = float(event.get("timestamp", 0))
        if timestamp / FRAME_MS > span:
            continue
        placed.append((event, pid, puuid, timestamp))
    if not placed:
        return []
    masses, mix = ward_posterior(parsed, [(puuid, timestamp) for _, _, puuid, timestamp in placed], sigma)
    spots = np.array([_next_gank(jungler_moves[parsed.teams.get(pid, 100)][1], timestamp) for _, pid, _, timestamp in placed])
    ganked = ~np.isnan(spots[:, 0])
    near = np.zeros(len(placed))
    near[ganked] = radius_mass({key: values[ganked] for key, values in mix.items()}, spots[ganked], GANK_RADIUS)

    rows = []
    for (event, pid, puuid, timestamp), mass, chance in zip(placed, masses, near):
        if mass.sum() == 0.0:
            continue
        team = parsed.teams.get(pid, 100)
        invades, _ = jungler_moves[team]
        next_invade = min(
            (moment - timestamp for moment in invades if moment >= timestamp), default=None
        )
        own_lane = lane_states.get(pid)
        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "role": parsed.roles.get(pid, "") or "UNKNOWN",
                "team_id": team,
                "timestamp_ms": int(timestamp),
                "clock": f"{int(timestamp // 60000)}:{int(timestamp % 60000 // 1000):02d}",
                "minute": timestamp / FRAME_MS,
                "ward_type": event.get("wardType"),
                **{column: float(value) for column, value in zip(ZONE_COLUMNS, mass)},
                "own_lane_state": own_lane[int(timestamp // 60000)]
                if own_lane and int(timestamp // 60000) < len(own_lane)
                else None,
                "seconds_before_jungler_invade": None
                if next_invade is None
                else round(next_invade / 1000.0, 1),
                "warded_before_invade": bool(next_invade is not None and next_invade <= INVADE_LEAD),
                "warded_near_next_gank": float(chance),
            }
        )
    return rows
