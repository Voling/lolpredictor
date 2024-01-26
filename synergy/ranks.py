from .config import TIER_ORDER

TIER_BASE_LP = {
    "IRON": 0,
    "BRONZE": 400,
    "SILVER": 800,
    "GOLD": 1200,
    "PLATINUM": 1600,
    "EMERALD": 2000,
    "DIAMOND": 2400,
    "MASTER": 2800,
    "GRANDMASTER": 2800,
    "CHALLENGER": 2800,
}

DIVISION_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4}


def division_to_int(division: str | None) -> int:
    if not division:
        return 1
    return DIVISION_TO_INT.get(division.upper(), 1)


def rank_to_lp(tier: str | None, division: str | None, lp: int | None) -> int:
    if not tier:
        return 0
    base = TIER_BASE_LP.get(tier.upper())
    if base is None:
        return 0
    if base >= TIER_BASE_LP["MASTER"]:
        return base + int(lp or 0)
    steps = 4 - division_to_int(division)
    return base + steps * 100 + int(lp or 0)


def tier_index(tier: str | None) -> int:
    if not tier:
        return -1
    try:
        return TIER_ORDER.index(tier.upper())
    except ValueError:
        return -1


def pick_solo_entry(entries: list[dict]) -> dict | None:
    for entry in entries or []:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            return entry
    return None
