import math

REGIONS = [
    "LANE_TOP_OWN",
    "LANE_TOP_NEUTRAL",
    "LANE_TOP_ENEMY",
    "LANE_MID_OWN",
    "LANE_MID_NEUTRAL",
    "LANE_MID_ENEMY",
    "LANE_BOT_OWN",
    "LANE_BOT_NEUTRAL",
    "LANE_BOT_ENEMY",
    "JUNGLE_OWN_TOPSIDE",
    "JUNGLE_OWN_BOTSIDE",
    "JUNGLE_ENEMY_TOPSIDE",
    "JUNGLE_ENEMY_BOTSIDE",
    "RIVER_BARON",
    "RIVER_DRAGON",
    "BASE_OWN",
    "BASE_ENEMY",
    "UNKNOWN",
]
REGION_INDEX = {name: index for index, name in enumerate(REGIONS)}
MAP_SPAN = 15000.0
LANE_TOLERANCE = 1500.0
RIVER_TOLERANCE = 900.0
BASE_RADIUS = 3200.0


def _depth(progress: float, team: int) -> str:
    if team == 200:
        progress = 2.0 - progress
    if progress < 0.85:
        return "OWN"
    if progress < 1.15:
        return "NEUTRAL"
    return "ENEMY"


def region_of(x: float, y: float, team: int) -> int:
    if x <= 0 and y <= 0:
        return REGION_INDEX["UNKNOWN"]
    diagonal = x - y
    progress = (x + y) / MAP_SPAN
    if math.hypot(x, y) < BASE_RADIUS:
        return REGION_INDEX["BASE_OWN" if team == 100 else "BASE_ENEMY"]
    if math.hypot(MAP_SPAN - x, MAP_SPAN - y) < BASE_RADIUS:
        return REGION_INDEX["BASE_ENEMY" if team == 100 else "BASE_OWN"]
    depth = _depth(progress, team)
    if abs(diagonal) / math.sqrt(2) < LANE_TOLERANCE:
        return REGION_INDEX[f"LANE_MID_{depth}"]
    if (x < 3200 and y > 3200) or (y > 11800 and x < 11800):
        return REGION_INDEX[f"LANE_TOP_{depth}"]
    if (y < 3200 and x > 3200) or (x > 11800 and y < 11800):
        return REGION_INDEX[f"LANE_BOT_{depth}"]
    if abs(x + y - MAP_SPAN) / math.sqrt(2) < RIVER_TOLERANCE:
        return REGION_INDEX["RIVER_BARON" if diagonal < 0 else "RIVER_DRAGON"]
    side = "TOPSIDE" if diagonal < 0 else "BOTSIDE"
    own_half = (x + y) < MAP_SPAN if team == 100 else (x + y) >= MAP_SPAN
    return REGION_INDEX[f"JUNGLE_{'OWN' if own_half else 'ENEMY'}_{side}"]


def regions_of(x, y, team):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    team = np.asarray(team)
    out = np.full(x.shape, REGION_INDEX["UNKNOWN"], dtype=np.int16)
    todo = ~((x <= 0) & (y <= 0))

    blue = team == 100
    diagonal = x - y
    progress = (x + y) / MAP_SPAN
    shifted = np.where(blue, progress, 2.0 - progress)
    depth = np.where(shifted < 0.85, "OWN", np.where(shifted < 1.15, "NEUTRAL", "ENEMY"))

    near = np.hypot(x, y) < BASE_RADIUS
    far = np.hypot(MAP_SPAN - x, MAP_SPAN - y) < BASE_RADIUS
    for hit, label in (
        (todo & near & blue, "BASE_OWN"),
        (todo & near & ~blue, "BASE_ENEMY"),
        (todo & ~near & far & blue, "BASE_ENEMY"),
        (todo & ~near & far & ~blue, "BASE_OWN"),
    ):
        out[hit] = REGION_INDEX[label]
    todo &= ~(near | far)

    root = math.sqrt(2.0)
    lanes = {
        "LANE_MID": np.abs(diagonal) / root < LANE_TOLERANCE,
        "LANE_TOP": ((x < 3200) & (y > 3200)) | ((y > 11800) & (x < 11800)),
        "LANE_BOT": ((y < 3200) & (x > 3200)) | ((x > 11800) & (y < 11800)),
    }
    for lane, mask in lanes.items():
        for name in ("OWN", "NEUTRAL", "ENEMY"):
            hit = todo & mask & (depth == name)
            out[hit] = REGION_INDEX[f"{lane}_{name}"]
        todo &= ~mask

    river = np.abs(x + y - MAP_SPAN) / root < RIVER_TOLERANCE
    out[todo & river & (diagonal < 0)] = REGION_INDEX["RIVER_BARON"]
    out[todo & river & (diagonal >= 0)] = REGION_INDEX["RIVER_DRAGON"]
    todo &= ~river

    own = np.where(blue, (x + y) < MAP_SPAN, (x + y) >= MAP_SPAN)
    for half, side in (("OWN", "TOPSIDE"), ("OWN", "BOTSIDE"), ("ENEMY", "TOPSIDE"), ("ENEMY", "BOTSIDE")):
        hit = todo & (own == (half == "OWN")) & ((diagonal < 0) == (side == "TOPSIDE"))
        out[hit] = REGION_INDEX[f"JUNGLE_{half}_{side}"]
    return out
