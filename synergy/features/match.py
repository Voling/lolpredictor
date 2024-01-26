from typing import Any

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return float(default if value is None else int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position(participant: dict) -> str:
    raw = (participant.get("teamPosition") or participant.get("individualPosition") or "").upper()
    return raw if raw in POSITIONS else "UNKNOWN"


def match_duration_minutes(info: dict) -> float:
    duration = _num(info.get("gameDuration"))
    if duration > 20000:
        duration = duration / 1000.0
    return max(duration / 60.0, 1.0)


def participant_rows(match: dict) -> list[dict]:
    info = match["info"]
    match_id = match["metadata"]["matchId"]
    minutes = match_duration_minutes(info)
    patch = ".".join(str(info.get("gameVersion", "")).split(".")[:2])
    return [
        {
            "match_id": match_id,
            "puuid": participant["puuid"],
            "team_id": int(participant.get("teamId", 0)),
            "position": _position(participant),
            "champion_id": int(_num(participant.get("championId"))),
            "champion_name": participant.get("championName", ""),
            "win": int(bool(participant.get("win"))),
            "patch": patch,
            "game_creation": int(_num(info.get("gameCreation"))),
            "duration_min": minutes,
            "surrendered": int(bool(participant.get("gameEndedInSurrender"))),
            "early_surrender": int(bool(participant.get("gameEndedInEarlySurrender"))),
        }
        for participant in info["participants"]
    ]
