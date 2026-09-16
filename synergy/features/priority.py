from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..chunks import ChunkWriter, merge
from ..config import Settings, get_settings
from ..deep.stream import SEATS, _by_match, _seat_points, event_rows, frame_tracks, seat_table
from ..ingest.store import Store
from .cells import cell_block, evidence_share
from .posterior import bridge_at, dead_mask, load_sigma, progress_moments, region_mass, top_regions, wave_mass
from .regions import REGION_INDEX
from .wave import LANE_PREFIX, WAVE_CS

TICK = 10.0 / 60.0
TICKS = np.round(np.arange(0.0, 15.0, TICK), 6)
OBJECTIVE_WINDOW = 1.0
LANES = {"LANE_TOP": ("TOP",), "LANE_MID": ("MIDDLE",), "LANE_BOT": ("BOTTOM", "UTILITY")}
BAND_EDGES = np.array([0.85, 0.95, 1.05, 1.15])
BANDS = ("deep_own", "own", "mid", "theirs", "deep_theirs")
OPPONENT = (*BANDS, "absent")
FARMING = ("farm", "idle")
OUTCOMES = [f"{own}_{opp}_{farm}" for own in BANDS for opp in OPPONENT for farm in FARMING] + ["off_lane", "dead"]
SITUATIONS = ["all", "pre_objective"]
PRIORITY_COLUMNS = [f"prio_{s}_{o}" for s in SITUATIONS for o in OUTCOMES]
TRACK = "positions_10s.parquet"
TABLE = "priority.parquet"
EVIDENCE = "evidence_priority.parquet"
WORKERS = 8
CHUNK_MATCHES = 400
_zones = {
    prefix: np.array([REGION_INDEX[f"{prefix}_{depth}"] for depth in ("OWN", "NEUTRAL", "ENEMY")])
    for prefix in LANES
}


def lane_priority(
    u_mine: np.ndarray, var_mine: np.ndarray, presence_mine: np.ndarray,
    u_theirs: np.ndarray, var_theirs: np.ndarray, presence_theirs: np.ndarray, blue: bool,
) -> tuple[np.ndarray, np.ndarray]:
    front = (u_mine + u_theirs) / 2.0
    spread = np.sqrt((var_mine + var_theirs) / 4.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (1.0 - front) / np.where(spread > 0, spread, 1.0)
        past = np.where(spread > 0, 1.0 - norm.cdf(z), front >= 1.0)
    ahead = past if blue else 1.0 - past
    ahead = np.where(np.isfinite(ahead), ahead, 0.0)
    prio = presence_mine * ((1.0 - presence_theirs) + presence_theirs * ahead)
    gap = np.where(blue, front - 1.0, 1.0 - front)
    return np.clip(prio, 0.0, 1.0), gap


def band_weights(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(var, 0.0, None))
    edges = np.concatenate([[-np.inf], BAND_EDGES, [np.inf]])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (edges[None, :] - mean[:, None]) / np.where(sd[:, None] > 0, sd[:, None], 1.0)
        cdf = np.where(sd[:, None] > 0, norm.cdf(z), (edges[None, :] > mean[:, None]).astype(float))
    cdf[:, 0], cdf[:, -1] = 0.0, 1.0
    return np.diff(cdf, axis=1)


def _side(u: np.ndarray, var: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = weight.sum(axis=0)
    safe = np.where(total > 0, total, 1.0)
    mean = (weight * u).sum(axis=0) / safe
    spread = (weight**2 * var).sum(axis=0) / safe**2
    return mean, spread, np.clip(weight.max(axis=0), 0.0, 1.0)


def _seat_pass(points, spans, spot, blue_of, roles, sigma, filtered: bool) -> dict:
    ticks = len(TICKS)
    out = {
        "u_mean": np.zeros((SEATS, ticks)), "u_var": np.zeros((SEATS, ticks)),
        "in_lane": np.zeros((SEATS, ticks)), "alive": np.zeros((SEATS, ticks), bool),
        "push": np.zeros((SEATS, ticks)), "defensive": np.zeros((SEATS, ticks)),
        "region": np.zeros((SEATS, ticks), np.int8), "region_p": np.zeros((SEATS, ticks), np.float16),
        "farmed": np.zeros((SEATS, ticks)),
    }
    floor = np.clip(TICKS.astype(int), 0, spot["cs"].shape[1] - 1)
    for seat in range(SEATS):
        team = 100 if blue_of[seat] else 200
        role = roles[seat]
        mix = bridge_at(points[seat], TICKS, sigma.get(role, sigma["TOP"]), filtered=filtered)
        dead = dead_mask(spans[seat], TICKS)
        masses = region_mass(mix, team)
        masses[dead] = 0.0
        mean, var, known = progress_moments(mix)
        live = known & ~dead & (masses.sum(axis=1) > 0)
        out["alive"][seat] = live
        out["u_mean"][seat] = np.where(live, mean, 0.0)
        out["u_var"][seat] = np.where(live, var, 0.0)
        prefix = LANE_PREFIX.get(role)
        if prefix is not None:
            out["in_lane"][seat] = np.where(live, masses[:, _zones[prefix]].sum(axis=1), 0.0)
        cs = spot["cs"][seat]
        farmed = np.where(floor > 0, cs[floor] - cs[np.clip(floor - 1, 0, None)], 0.0)
        out["farmed"][seat] = farmed
        waves = wave_mass(masses, mix, team, role, farmed, ~live)
        out["push"][seat], out["defensive"][seat] = waves[:, 0], waves[:, 1]
        order, picked = top_regions(masses, top=1)
        out["region"][seat] = np.where(live, order[:, 0] + 1, 0)
        out["region_p"][seat] = np.where(live, picked[:, 0], 0.0)
    return out


def _lane_pass(seat: dict, blue_of, roles) -> dict:
    ticks = len(TICKS)
    out = {
        "prio": np.full((SEATS, ticks), np.nan), "gap": np.full((SEATS, ticks), np.nan),
        "opp_u": np.zeros((SEATS, ticks)), "opp_var": np.zeros((SEATS, ticks)), "opp_presence": np.zeros((SEATS, ticks)),
    }
    for prefix, lane_roles in LANES.items():
        for side in (1, 0):
            mine = [s for s in range(SEATS) if blue_of[s] == side and roles[s] in lane_roles]
            theirs = [s for s in range(SEATS) if blue_of[s] != side and roles[s] in lane_roles]
            if not mine or not theirs:
                continue
            um, vm, pm = _side(seat["u_mean"][mine], seat["u_var"][mine], seat["in_lane"][mine])
            ut, vt, pt = _side(seat["u_mean"][theirs], seat["u_var"][theirs], seat["in_lane"][theirs])
            lane_prio, lane_gap = lane_priority(um, vm, pm, ut, vt, pt, blue=bool(side))
            for s in mine:
                out["prio"][s] = np.where(seat["alive"][s], lane_prio, np.nan)
                out["gap"][s] = np.where(seat["alive"][s] & (pm > 0) & (pt > 0), lane_gap, np.nan)
                out["opp_u"][s], out["opp_var"][s], out["opp_presence"][s] = ut, vt, pt
    return out


def lane_counts(seat: dict, lane: dict, blue_of, roles, objective_ticks: np.ndarray) -> list[tuple]:
    rows = []
    for s in range(SEATS):
        if LANE_PREFIX.get(roles[s]) is None:
            continue
        flip = blue_of[s] == 0
        own_mean = 2.0 - seat["u_mean"][s] if flip else seat["u_mean"][s]
        opp_mean = 2.0 - lane["opp_u"][s] if flip else lane["opp_u"][s]
        own = band_weights(own_mean, seat["u_var"][s])
        opp = band_weights(opp_mean, lane["opp_var"][s])
        presence = lane["opp_presence"][s][:, None]
        opp = np.concatenate([opp * presence, 1.0 - presence], axis=1)
        farm = np.stack([seat["farmed"][s] >= WAVE_CS, seat["farmed"][s] < WAVE_CS], axis=1).astype(float)
        joint = own[:, :, None, None] * opp[:, None, :, None] * farm[:, None, None, :]
        in_lane = seat["in_lane"][s][:, None, None, None]
        alive = seat["alive"][s]
        weights = np.concatenate(
            [(joint * in_lane).reshape(len(TICKS), -1), (1.0 - seat["in_lane"][s])[:, None], (~alive)[:, None].astype(float)], axis=1
        )
        weights[~alive, :-1] = 0.0
        for situation, mask in (("all", np.ones(len(TICKS), bool)), ("pre_objective", objective_ticks)):
            totals = weights[mask].sum(axis=0)
            rows.extend((s, situation, outcome, float(total)) for outcome, total in zip(OUTCOMES, totals) if total > 0)
    return rows


def match_priority(match: dict, spot: dict, events: list[tuple], sigma: dict) -> tuple[list[dict], list[dict]]:
    seat_of = {puuid: index + 1 for index, puuid in enumerate(match["puuid"])}
    blue_of = [1 if team == 100 else 0 for team in match["team"]]
    roles = match["position"]
    points, spans = _seat_points(match, spot, events)
    smoothed = _seat_pass(points, spans, spot, blue_of, roles, sigma, filtered=False)
    lane = _lane_pass(smoothed, blue_of, roles)
    filtered = _lane_pass(_seat_pass(points, spans, spot, blue_of, roles, sigma, filtered=True), blue_of, roles)

    objective_ticks = np.zeros(len(TICKS), bool)
    for stamp, kind, *_ in events:
        if kind == "ELITE_MONSTER_KILL":
            when = float(stamp or 0) / 60000.0
            objective_ticks |= (TICKS >= when - OBJECTIVE_WINDOW) & (TICKS < when)

    track = []
    for seat in range(SEATS):
        puuid = match["puuid"][seat]
        for index in range(len(TICKS)):
            prio, held = lane["prio"][seat, index], filtered["prio"][seat, index]
            track.append({
                "match_id": match["match_id"], "puuid": puuid, "tick": float(TICKS[index]),
                "region": int(smoothed["region"][seat, index]) - 1, "region_p": float(smoothed["region_p"][seat, index]),
                "u_mean": float(smoothed["u_mean"][seat, index]), "u_sd": float(np.sqrt(smoothed["u_var"][seat, index])),
                "in_lane": float(smoothed["in_lane"][seat, index]),
                "prio": float(prio) if np.isfinite(prio) else None,
                "prio_filtered": float(held) if np.isfinite(held) else None,
                "push": float(smoothed["push"][seat, index]), "defensive": float(smoothed["defensive"][seat, index]),
                "alive": bool(smoothed["alive"][seat, index]),
            })
    counts = [
        {"match_id": match["match_id"], "puuid": match["puuid"][s], "situation": situation, "outcome": outcome, "count": count}
        for s, situation, outcome, count in lane_counts(smoothed, lane, blue_of, roles, objective_ticks)
    ]
    return track, counts


def _shard(settings: Settings, match_ids: list[str], shard: int) -> tuple[str, list[dict]]:
    seats = seat_table(settings, match_ids)
    wanted = sorted(seats)
    tracks = frame_tracks(settings, wanted, seats)
    grouped = _by_match(event_rows(settings, wanted))
    sigma = load_sigma(settings)
    shards = settings.processed_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    writer = ChunkWriter(shards / f"positions_10s.{shard:03d}.parquet")
    counts: list[dict] = []
    for match_id in wanted:
        spot = tracks.get(match_id)
        if spot is None:
            continue
        match = dict(seats[match_id], match_id=match_id)
        rows, found = match_priority(match, spot, grouped.get(match_id, []), sigma)
        writer.add(rows)
        counts.extend(found)
    writer.close()
    return str(writer.path), counts


def build_priority(settings: Settings | None = None, workers: int = WORKERS, limit: int | None = None) -> dict:
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
    chunks = [match_ids[start : start + CHUNK_MATCHES] for start in range(0, len(match_ids), CHUNK_MATCHES)]
    parts, counts = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, found in pool.map(_shard, [settings] * len(chunks), chunks, range(len(chunks))):
            parts.append(path)
            counts.extend(found)
    rows = merge([Path(p) for p in parts], settings.processed_dir / TRACK)
    for part in parts:
        Path(part).unlink(missing_ok=True)
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"])
    seats = seats[seats.match_id.isin(set(match_ids))]
    counted = pd.DataFrame(counts)
    table, report = cell_block(counted, seats, SITUATIONS, OUTCOMES, "prio")
    table.to_parquet(settings.processed_dir / TABLE, index=False)
    evidence = evidence_share(counted, seats, SITUATIONS, {situation: report[situation]["kappa"] for situation in SITUATIONS})
    evidence.to_parquet(settings.processed_dir / EVIDENCE, index=False)
    return {
        "matches": len(match_ids),
        "track_rows": int(rows),
        "player_matches": int(len(table)),
        "columns": len(PRIORITY_COLUMNS),
        "kappa": {situation: report[situation]["kappa"] for situation in SITUATIONS},
        "ticks_counted": {situation: round(report[situation]["rows"]) for situation in SITUATIONS},
        "evidence_share_median": round(float(evidence["share"].median()), 3),
    }


LANE_OF = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
CONTEXT_COLUMNS = ["lane_priority", "top_priority", "mid_priority", "bot_priority"]


def priority_context_from(track: pd.DataFrame, seats: pd.DataFrame) -> pd.DataFrame:
    ticks = track.dropna(subset=["prio_filtered"]).assign(minute=lambda f: np.floor(f["tick"]).astype(int))
    per_player = ticks.groupby(["match_id", "puuid", "minute"], as_index=False)["prio_filtered"].mean()
    per_player = per_player.merge(seats, on=["match_id", "puuid"])
    per_player["lane"] = per_player["position"].map(LANE_OF)
    lanes = (
        per_player.dropna(subset=["lane"])
        .groupby(["match_id", "team_id", "minute", "lane"])["prio_filtered"].mean()
        .unstack("lane").reindex(columns=["top", "mid", "bot"])
    )
    lanes.columns = [f"{lane}_priority" for lane in lanes.columns]
    lanes = lanes.reset_index()
    out = seats.merge(lanes, on=["match_id", "team_id"], how="inner")
    own = out["position"].map(LANE_OF)
    picked = np.select(
        [own == "top", own == "mid", own == "bot"],
        [out["top_priority"], out["mid_priority"], out["bot_priority"]],
        default=np.nan,
    )
    best = out[["top_priority", "mid_priority", "bot_priority"]].max(axis=1)
    out["lane_priority"] = pd.Series(picked, index=out.index).fillna(best)
    for column in ("top_priority", "mid_priority", "bot_priority"):
        out[column] = out[column].fillna(best)
    return out[["match_id", "puuid", "minute", *CONTEXT_COLUMNS]]


def priority_context(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TRACK
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", "minute", *CONTEXT_COLUMNS])
    track = pd.read_parquet(path, columns=["match_id", "puuid", "tick", "prio_filtered"])
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "team_id", "position"])
    return priority_context_from(track, seats)


def load_priority(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *PRIORITY_COLUMNS])
    table = pd.read_parquet(path)
    if any(column not in table.columns for column in PRIORITY_COLUMNS):
        return pd.DataFrame(columns=["match_id", "puuid", *PRIORITY_COLUMNS])
    return table
