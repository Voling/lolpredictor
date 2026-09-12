import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import Settings, get_settings
from .timeline import ParsedTimeline

SPAN = 15
STATE_COLUMNS = [
    "gold_diff",
    "xp_diff",
    "cs_diff",
    "kill_diff",
    "plate_diff",
    "turret_diff",
    "dragon_diff",
    "grub_diff",
]
GRUB = "HORDE"
DRAGON = "DRAGON"


def state_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    win = {
        int(p.get("teamId", 0)): int(bool(p.get("win")))
        for p in match["info"]["participants"]
    }
    running = {team: {"kills": 0, "plates": 0, "turrets": 0, "dragons": 0, "grubs": 0}
               for team in (100, 200)}
    schedule: dict[int, list[tuple[int, str]]] = {}
    for event in parsed.events:
        minute = int(float(event.get("timestamp", 0)) // 60000)
        if minute > limit:
            continue
        kind = event.get("type")
        actor = next((int(event[k]) for k in ("killerId", "creatorId") if event.get(k)), 0)
        team = parsed.teams.get(actor)
        if team is None:
            continue
        if kind == "CHAMPION_KILL":
            field = "kills"
        elif kind == "TURRET_PLATE_DESTROYED":
            field = "plates"
        elif kind == "BUILDING_KILL":
            field = "turrets"
        elif kind == "ELITE_MONSTER_KILL":
            monster = str(event.get("monsterType") or "")
            field = "dragons" if monster == DRAGON else "grubs" if monster == GRUB else ""
        else:
            field = ""
        if field:
            schedule.setdefault(minute, []).append((team, field))
    rows = []
    for minute in range(limit + 1):
        for team, field in schedule.get(minute, []):
            running[team][field] += 1
        totals = {}
        for team in (100, 200):
            members = [pid for pid, value in parsed.teams.items() if value == team]
            totals[team] = {
                "gold": sum(parsed.at_minute(parsed.gold, pid, minute) for pid in members),
                "xp": sum(parsed.at_minute(parsed.xp, pid, minute) for pid in members),
                "cs": sum(parsed.at_minute(parsed.cs, pid, minute) for pid in members),
            }
        for team in (100, 200):
            enemy = 200 if team == 100 else 100
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "team_id": team,
                    "minute": minute,
                    "win": win.get(team, 0),
                    "gold_diff": (totals[team]["gold"] - totals[enemy]["gold"]) / 1000.0,
                    "xp_diff": (totals[team]["xp"] - totals[enemy]["xp"]) / 1000.0,
                    "cs_diff": float(totals[team]["cs"] - totals[enemy]["cs"]),
                    "kill_diff": float(running[team]["kills"] - running[enemy]["kills"]),
                    "plate_diff": float(running[team]["plates"] - running[enemy]["plates"]),
                    "turret_diff": float(running[team]["turrets"] - running[enemy]["turrets"]),
                    "dragon_diff": float(running[team]["dragons"] - running[enemy]["dragons"]),
                    "grub_diff": float(running[team]["grubs"] - running[enemy]["grubs"]),
                }
            )
    return rows


def fit_evaluation(states: pd.DataFrame, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    frame = states.dropna(subset=STATE_COLUMNS + ["win"])
    if len(frame) < 2000:
        return {}
    design = frame[STATE_COLUMNS].to_numpy(dtype=float)
    minutes = frame["minute"].to_numpy(dtype=float).reshape(-1, 1)
    matrix = np.hstack([design, design * minutes / 15.0, minutes / 15.0])
    model = LogisticRegression(max_iter=3000, C=1.0).fit(matrix, frame["win"].astype(int))
    weights = dict(zip(
        STATE_COLUMNS + [f"{name}_x_minute" for name in STATE_COLUMNS] + ["minute"],
        [round(float(value), 5) for value in model.coef_[0]],
    ))
    payload = {
        "weights": weights,
        "intercept": round(float(model.intercept_[0]), 5),
        "rows": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_dir / "evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def evaluate(states: pd.DataFrame, model: dict) -> np.ndarray:
    weights = model["weights"]
    minutes = states["minute"].to_numpy(dtype=float) / 15.0
    total = np.full(len(states), float(model["intercept"])) + weights["minute"] * minutes
    for name in STATE_COLUMNS:
        values = states[name].to_numpy(dtype=float)
        total += weights[name] * values + weights[f"{name}_x_minute"] * values * minutes
    return 1.0 / (1.0 + np.exp(-total))


def load_evaluation(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.processed_dir / "evaluation.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
