import json
import math

import numpy as np
from scipy.stats import norm

from ..config import Settings, get_settings
from .anchors import FOUNTAIN, PLACE_EVENTS, SHOP_EVENTS, plausible, respawn_delay
from .regions import MAP_SPAN, REGION_INDEX, REGIONS, regions_of
from .wave import DEFENSIVE_PROGRESS, LANE_PREFIX, PUSH_PROGRESS, WAVE_CS

GRID = 48
TAU_CENTRES = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
DEFAULT_TABLE = [600.0, 1250.0, 1650.0, 2000.0, 2450.0]
SIGMA_FILE = "position_sigma.json"
TOP = 3
CHUNK = 4096
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
MIN_PER_BIN = 100

_centres = (np.arange(GRID) + 0.5) * MAP_SPAN / GRID
_cell_x, _cell_y = (axis.ravel() for axis in np.meshgrid(_centres, _centres, indexing="xy"))
_cell_region = {
    team: regions_of(_cell_x, _cell_y, np.full(_cell_x.shape, team)) for team in (100, 200)
}
_membership = {
    team: np.eye(len(REGIONS), dtype=np.float32)[_cell_region[team]] for team in (100, 200)
}
_lane_zones = {
    prefix: np.array([REGION_INDEX[f"{prefix}_{depth}"] for depth in ("OWN", "NEUTRAL", "ENEMY")])
    for prefix in set(LANE_PREFIX.values())
}


def load_sigma(settings: Settings | None = None) -> dict[str, list[float]]:
    settings = settings or get_settings()
    path = settings.model_dir / SIGMA_FILE
    if not path.exists():
        return {role: list(DEFAULT_TABLE) for role in ROLES}
    with open(path, encoding="utf-8") as handle:
        return {role: [float(v) for v in values] for role, values in json.load(handle)["sigma"].items()}


def sigma_at(table: list[float], tau: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=float)
    tau = np.asarray(tau, dtype=float)
    inside = np.interp(tau, TAU_CENTRES, table)
    slope = (table[-1] - table[-2]) / (TAU_CENTRES[-1] - TAU_CENTRES[-2])
    beyond = table[-1] + slope * (tau - TAU_CENTRES[-1])
    return np.where(tau > TAU_CENTRES[-1], beyond, inside)


def death_spans(kills: list[tuple[float, int]]) -> list[tuple[float, float]]:
    return sorted((when, when + respawn_delay(level)) for when, level in kills)


def dead_mask(spans: list[tuple[float, float]], minutes: np.ndarray) -> np.ndarray:
    dead = np.zeros(len(minutes), bool)
    for start, end in spans:
        dead |= (minutes >= start) & (minutes < end)
    return dead


def known_points(
    frames: np.ndarray,
    known: np.ndarray,
    spans: list[tuple[float, float]],
    certain: list[tuple[float, float, float]],
    claimed: list[tuple[float, float, float]],
    team: int,
) -> np.ndarray:
    minutes = np.arange(len(frames), dtype=float)
    alive = known & ~dead_mask(spans, minutes)
    points = [(float(minute), float(frames[minute, 0]), float(frames[minute, 1])) for minute in np.flatnonzero(alive)]
    points.extend(certain)
    fountain = FOUNTAIN[team]
    points.extend((end, fountain[0], fountain[1]) for _, end in spans)
    points.sort()
    for when, x, y in claimed:
        if plausible(points, when, (x, y)):
            points.append((when, x, y))
    points.sort()
    return np.array(points, dtype=float).reshape(-1, 3)


def bracket(points: np.ndarray, minutes: np.ndarray) -> dict:
    minutes = np.asarray(minutes, dtype=float)
    n = len(minutes)
    out = {
        "p0": np.zeros((n, 2)), "p1": np.zeros((n, 2)),
        "tau0": np.full(n, np.inf), "tau1": np.full(n, np.inf),
        "left": np.zeros(n, bool), "right": np.zeros(n, bool), "exact": np.zeros(n, bool),
    }
    if len(points) == 0:
        return out
    times = points[:, 0]
    right = np.searchsorted(times, minutes, side="left")
    clipped = np.clip(right, 0, len(points) - 1)
    exact = (right < len(points)) & (times[clipped] == minutes)
    left = right - 1
    has_left = left >= 0
    has_right = (right < len(points)) & ~exact
    out["exact"] = exact
    out["p1"][exact] = points[clipped[exact], 1:]
    out["left"], out["right"] = has_left & ~exact, has_right
    lo = np.clip(left, 0, len(points) - 1)
    out["p0"][has_left] = points[lo[has_left], 1:]
    out["tau0"][has_left] = minutes[has_left] - times[lo[has_left]]
    out["p1"][has_right] = points[clipped[has_right], 1:]
    out["tau1"][has_right] = times[clipped[has_right]] - minutes[has_right]
    return out


def bridge_at(points: np.ndarray, minutes: np.ndarray, table: list[float], filtered: bool = False) -> dict:
    span = bracket(points, minutes)
    if filtered:
        span["right"] = np.zeros_like(span["right"])
    n = len(span["exact"])
    weight = np.zeros(n)
    s0 = np.full(n, np.inf)
    s1 = np.full(n, np.inf)
    both = span["left"] & span["right"]
    weight[both] = span["tau0"][both] / (span["tau0"][both] + span["tau1"][both])
    weight[span["right"] & ~span["left"]] = 1.0
    s0[span["left"]] = sigma_at(table, span["tau0"][span["left"]])
    s1[span["right"]] = sigma_at(table, span["tau1"][span["right"]])
    exact = span["exact"]
    weight[exact], s1[exact] = 1.0, 0.0
    return {"p0": span["p0"], "p1": span["p1"], "w": weight, "s0": s0, "s1": s1}


def _single_mass(mean: np.ndarray, spread: np.ndarray, team: int) -> np.ndarray:
    out = np.zeros((len(mean), len(REGIONS)), np.float32)
    cell = MAP_SPAN / GRID
    finite = np.isfinite(spread)
    sharp = finite & (spread < cell / 2.0)
    if sharp.any():
        hit = regions_of(mean[sharp, 0], mean[sharp, 1], np.full(int(sharp.sum()), team))
        out[np.flatnonzero(sharp), hit] = 1.0
    soft = np.flatnonzero(finite & ~sharp)
    for start in range(0, len(soft), CHUNK):
        rows = soft[start : start + CHUNK]
        dx = _cell_x[None, :] - mean[rows, 0, None]
        dy = _cell_y[None, :] - mean[rows, 1, None]
        density = np.exp(-(dx * dx + dy * dy) / (2.0 * spread[rows, None] ** 2)).astype(np.float32)
        total = density.sum(axis=1, keepdims=True)
        density /= np.where(total > 0, total, 1.0)
        out[rows] = density @ _membership[team]
    return out


def region_mass(mix: dict, team: int) -> np.ndarray:
    w = mix["w"][:, None].astype(np.float32)
    return (1.0 - w) * _single_mass(mix["p0"], mix["s0"], team) + w * _single_mass(mix["p1"], mix["s1"], team)


def progress_moments(mix: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = mix["w"]
    u0 = (mix["p0"][:, 0] + mix["p0"][:, 1]) / MAP_SPAN
    u1 = (mix["p1"][:, 0] + mix["p1"][:, 1]) / MAP_SPAN
    left_known = (w < 1.0) & np.isfinite(mix["s0"])
    right_known = (w > 0.0) & np.isfinite(mix["s1"])
    v0 = np.where(left_known, 2.0 * np.where(left_known, mix["s0"], 0.0) ** 2 / MAP_SPAN**2, 0.0)
    v1 = np.where(right_known, 2.0 * np.where(right_known, mix["s1"], 0.0) ** 2 / MAP_SPAN**2, 0.0)
    mean = (1.0 - w) * u0 + w * u1
    second = (1.0 - w) * (v0 + u0**2) + w * (v1 + u1**2)
    known = left_known | right_known
    return mean, np.clip(second - mean**2, 0.0, None), known


def _progress_tail(mean: np.ndarray, spread: np.ndarray, team: int, threshold: float, above: bool) -> np.ndarray:
    progress = (mean[:, 0] + mean[:, 1]) / MAP_SPAN
    if team == 200:
        progress = 2.0 - progress
    scale = spread * math.sqrt(2.0) / MAP_SPAN
    finite = np.isfinite(scale)
    out = np.zeros(len(mean))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (threshold - progress) / np.where(scale > 0, scale, 1.0)
        soft = 1.0 - norm.cdf(z) if above else norm.cdf(z)
        hard = (progress >= threshold) if above else (progress <= threshold)
        out[finite] = np.where(scale[finite] > 0, soft[finite], hard[finite])
    return out


def wave_mass(
    masses: np.ndarray, mix: dict, team: int, role: str, lane_cs: np.ndarray, dead: np.ndarray
) -> np.ndarray:
    out = np.zeros((len(masses), 3), np.float32)
    prefix = LANE_PREFIX.get(role)
    if prefix is None:
        out[:, 2] = 1.0
        return out
    in_lane = masses[:, _lane_zones[prefix]].sum(axis=1)
    w = mix["w"]
    pushed = (1.0 - w) * _progress_tail(mix["p0"], mix["s0"], team, PUSH_PROGRESS, True) + w * _progress_tail(
        mix["p1"], mix["s1"], team, PUSH_PROGRESS, True
    )
    held = (1.0 - w) * _progress_tail(mix["p0"], mix["s0"], team, DEFENSIVE_PROGRESS, False) + w * _progress_tail(
        mix["p1"], mix["s1"], team, DEFENSIVE_PROGRESS, False
    )
    farmed = (lane_cs >= WAVE_CS).astype(np.float32)
    out[:, 0] = in_lane * pushed * farmed
    out[:, 1] = in_lane * held * farmed
    out[:, 2] = 1.0 - in_lane
    out[dead] = (0.0, 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def top_regions(masses: np.ndarray, top: int = TOP) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-masses, axis=1)[:, :top]
    picked = np.take_along_axis(masses, order, axis=1)
    return order.astype(np.int8), picked.astype(np.float16)


def anchor_points(
    events: list[tuple], seat_of: dict[str, int], blue_of: list[int]
) -> tuple[dict[int, list], dict[int, list], dict[int, list]]:
    certain: dict[int, list] = {}
    claimed: dict[int, list] = {}
    kills: dict[int, list] = {}
    for stamp, kind, actor, victim, assists, x, y, level in events:
        minute = float(stamp or 0) / 60000.0
        seat_actor, seat_victim = seat_of.get(actor, 0), seat_of.get(victim, 0)
        if kind == "CHAMPION_KILL":
            if seat_victim and x is not None:
                certain.setdefault(seat_victim, []).append((minute, float(x), float(y)))
                kills.setdefault(seat_victim, []).append((minute, int(level or 1)))
            for seat in (seat_actor, *(seat_of.get(p, 0) for p in (assists or []))):
                if seat and x is not None:
                    claimed.setdefault(seat, []).append((minute, float(x), float(y)))
        elif kind in SHOP_EVENTS and seat_actor:
            fountain = FOUNTAIN[100 if blue_of[seat_actor - 1] else 200]
            certain.setdefault(seat_actor, []).append((minute, fountain[0], fountain[1]))
        elif kind in PLACE_EVENTS and x is not None:
            for seat in (seat_actor, *(seat_of.get(p, 0) for p in (assists or []))):
                if seat:
                    claimed.setdefault(seat, []).append((minute, float(x), float(y)))
    return certain, claimed, kills


def nearer_residuals(points: np.ndarray, anchors: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    if not anchors or len(points) < 2:
        return []
    minutes = np.array([m for m, _, _ in anchors])
    spots = np.array([(x, y) for _, x, y in anchors])
    span = bracket(points, minutes)
    both = span["left"] & span["right"]
    d0 = np.hypot(*(spots - span["p0"]).T)
    d1 = np.hypot(*(spots - span["p1"]).T)
    nearer_is_left = d0 <= d1
    tau = np.where(nearer_is_left, span["tau0"], span["tau1"])
    squared = np.where(nearer_is_left, d0, d1) ** 2
    return list(zip(tau[both].tolist(), squared[both].tolist()))


def fit_sigma(residuals: dict[str, list[tuple[float, float]]]) -> dict[str, list[float]]:
    edges = np.concatenate([[0.0], (TAU_CENTRES[:-1] + TAU_CENTRES[1:]) / 2.0, [np.inf]])
    out = {}
    for role in ROLES:
        rows = np.array(residuals.get(role, []), dtype=float).reshape(-1, 2)
        table = []
        for index in range(len(TAU_CENTRES)):
            pick = (rows[:, 0] >= edges[index]) & (rows[:, 0] < edges[index + 1]) if len(rows) else np.zeros(0, bool)
            if pick.sum() < MIN_PER_BIN:
                table.append(DEFAULT_TABLE[index])
            else:
                table.append(float(math.sqrt(rows[pick, 1].mean() / 2.0)))
        out[role] = table
    return out
