import math

MAX_SPEED = 787.0
MAX_REACH = 12772.0
FRAME_SECONDS = 60.0
FOUNTAIN = {100: (394.0, 461.0), 200: (14340.0, 14454.0)}
SHOP_EVENTS = {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO"}
PLACE_EVENTS = {
    "TURRET_PLATE_DESTROYED",
    "ELITE_MONSTER_KILL",
    "CHAMPION_SPECIAL_KILL",
    "BUILDING_KILL",
}
RESPAWN_SECONDS = {
    1: 10.0, 2: 10.0, 3: 12.0, 4: 12.0, 5: 14.0, 6: 16.0, 7: 20.0, 8: 25.0, 9: 28.0,
    10: 32.5, 11: 35.0, 12: 37.5, 13: 40.0, 14: 42.5, 15: 45.0, 16: 47.5, 17: 50.0, 18: 52.5,
}


def respawn_delay(level: int) -> float:
    return RESPAWN_SECONDS.get(max(1, min(int(level), 18)), 10.0) / 60.0


def death_windows(parsed, limit: float) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for event in parsed.events:
        if event.get("type") != "CHAMPION_KILL":
            continue
        pid = int(event.get("victimId", 0))
        if not pid or pid not in parsed.pid_to_puuid:
            continue
        when = float(event.get("timestamp", 0)) / 60000.0
        if when > limit:
            continue
        levels = parsed.levels.get(pid) or []
        index = min(int(when), len(levels) - 1) if levels else -1
        level = levels[index] if index >= 0 else 1
        out.setdefault(pid, []).append((when, when + respawn_delay(level)))
    for values in out.values():
        values.sort()
    return out


def dead_at(windows: list[tuple[float, float]], minute: float) -> bool:
    return any(start <= minute < end for start, end in windows)


def _within(frame: tuple[float, float], spot: tuple[float, float], seconds: float) -> bool:
    allowed = min(MAX_SPEED * max(seconds, 1.0), MAX_REACH)
    return math.hypot(frame[0] - spot[0], frame[1] - spot[1]) <= allowed


def plausible(
    certainties: list[tuple[float, float, float]], minute: float, spot: tuple[float, float]
) -> bool:
    before = [c for c in certainties if c[0] <= minute]
    after = [c for c in certainties if c[0] >= minute]
    if not before or not after:
        return False
    last = max(before)
    first = min(after)
    return _within((last[1], last[2]), spot, (minute - last[0]) * FRAME_SECONDS) and _within(
        (first[1], first[2]), spot, (first[0] - minute) * FRAME_SECONDS
    )


def _claims(event: dict) -> list[tuple[str, int]]:
    kind = event.get("type")
    if kind == "CHAMPION_KILL":
        out = [("certain", int(event.get("victimId", 0))), ("claimed", int(event.get("killerId", 0)))]
        return out + [("claimed", int(p)) for p in (event.get("assistingParticipantIds") or [])]
    if kind in SHOP_EVENTS:
        return [("shop", int(event.get("participantId", 0)))]
    if kind in PLACE_EVENTS:
        out = [("claimed", int(event.get("killerId", 0)))]
        return out + [("claimed", int(p)) for p in (event.get("assistingParticipantIds") or [])]
    return []


def position_anchors(parsed, limit: float) -> dict[int, list[tuple[float, float, float, int]]]:
    windows = death_windows(parsed, limit)
    certain: dict[int, list[tuple[float, float, float]]] = {}
    claims: list[tuple[int, float, tuple[float, float], int]] = []
    for index, event in enumerate(parsed.events):
        minute = float(event.get("timestamp", 0)) / 60000.0
        if minute > limit:
            continue
        position = event.get("position") or {}
        for source, pid in _claims(event):
            if not pid or pid not in parsed.pid_to_puuid:
                continue
            if source == "shop":
                spot = FOUNTAIN.get(parsed.teams.get(pid, 100))
            elif "x" in position:
                spot = (float(position["x"]), float(position["y"]))
            else:
                continue
            if spot is None:
                continue
            if source == "claimed":
                claims.append((pid, minute, spot, index))
            else:
                certain.setdefault(pid, []).append((minute, spot[0], spot[1]))
    out: dict[int, list[tuple[float, float, float, int]]] = {}
    for pid in parsed.pid_to_puuid:
        track = parsed.positions.get(pid) or []
        fountain = FOUNTAIN.get(parsed.teams.get(pid, 100), (0.0, 0.0))
        points = list(certain.get(pid, ()))
        for _, respawn in windows.get(pid, ()):
            if respawn <= limit:
                points.append((respawn, fountain[0], fountain[1]))
        for index, (x, y) in enumerate(track):
            if index <= limit and not dead_at(windows.get(pid, ()), float(index)):
                points.append((float(index), x, y))
        points.sort()
        out[pid] = [(when, x, y, -1) for when, x, y in certain.get(pid, ())]
        for _, respawn in windows.get(pid, ()):
            if respawn <= limit:
                out[pid].append((respawn, fountain[0], fountain[1], -2))
        certain[pid] = points
    for pid, minute, spot, index in claims:
        if plausible(certain.get(pid, []), minute, spot):
            out.setdefault(pid, []).append((minute, spot[0], spot[1], index))
    for values in out.values():
        values.sort()
    return out


def confidence_at(
    anchors: list[tuple[float, float, float, int]],
    track: list[tuple[float, float]],
    minute: float,
) -> tuple[tuple[float, float], float]:
    best_spot, best_gap = None, None
    index = min(max(int(minute + 0.5), 0), max(len(track) - 1, 0))
    if track:
        best_spot, best_gap = track[index], abs(minute - index)
    for when, x, y, _ in anchors:
        gap = abs(minute - when)
        if best_gap is None or gap < best_gap:
            best_spot, best_gap = (x, y), gap
    if best_spot is None:
        return (0.0, 0.0), MAX_REACH
    radius = min(MAX_SPEED * best_gap * FRAME_SECONDS, MAX_REACH)
    return best_spot, radius
