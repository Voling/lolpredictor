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
