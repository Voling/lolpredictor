import json
import time

from .accounts import LinkError

IDS = "customs#ids"
MATCHES = "customs#match"
KIND = "tourney"
CUSTOM_GAME = "CUSTOM_GAME"
PLAYERS = 10
LIST_SECONDS = 6 * 3600
MATCH_SECONDS = 30 * 86_400
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


class Unavailable(Exception):
    pass


def together(seats: list[list], left: str, right: str) -> dict | None:
    mine = {seat[0]: seat for seat in seats}
    if left not in mine or right not in mine or len(seats) != PLAYERS:
        return None
    a, b = mine[left], mine[right]
    if a[1] != b[1] or a[2] == b[2]:
        return None
    enemies = {seat[2]: seat for seat in seats if seat[1] != a[1]}
    if a[2] not in enemies or b[2] not in enemies:
        return None
    return {
        "seats": [
            {"puuid": seat[0], "position": seat[2], "gold": float(seat[3]), "enemy": enemies[seat[2]][0], "enemy_gold": float(enemies[seat[2]][3])}
            for seat in (a, b)
        ]
    }


def seats_at(match: dict, timeline: dict, minute: int) -> list[list]:
    info = match.get("info") or {}
    participants = info.get("participants") or []
    frames = (timeline.get("info") or {}).get("frames") or []
    if len(participants) != PLAYERS or len(frames) <= minute:
        return []
    gold = frames[minute].get("participantFrames") or {}
    seats = []
    for participant in participants:
        position = participant.get("teamPosition") or ""
        frame = gold.get(str(participant.get("participantId")))
        if position not in POSITIONS or frame is None:
            return []
        seats.append([participant["puuid"], int(participant["teamId"]), position, float(frame.get("totalGold", 0.0))])
    return seats


class SharedCustoms:
    def __init__(self, riot, table, budget, minute: int, clock=time.time):
        self.riot, self.table, self.budget, self.minute, self.clock = riot, table, budget, minute, clock

    def between(self, left: str, right: str) -> tuple[list[dict], bool]:
        try:
            mine, theirs = self._ids(left), set(self._ids(right))
        except Unavailable:
            return [], False
        games = []
        for match_id in mine:
            if match_id not in theirs:
                continue
            try:
                seats = self._seats(match_id)
            except Unavailable:
                return games, False
            game = together(seats, left, right)
            if game:
                games.append({"match_id": match_id, **game})
        return games, True

    def _ids(self, puuid: str) -> list[str]:
        now = int(self.clock())
        found = self.table.get(IDS, puuid)
        if found and int(found.get("expires", 0)) > now:
            return json.loads(found["ids"])
        if not self.budget(1):
            raise Unavailable()
        try:
            ids = [str(match_id) for match_id in self.riot.match_ids(puuid, KIND)]
        except LinkError as exc:
            raise Unavailable() from exc
        self.table.put(IDS, puuid, {"ids": json.dumps(ids), "expires": now + LIST_SECONDS})
        return ids

    def _seats(self, match_id: str) -> list[list]:
        found = self.table.get(MATCHES, match_id)
        if found:
            return json.loads(found["seats"])
        if not self.budget(1):
            raise Unavailable()
        try:
            match = self.riot.match(match_id)
            usable = (match.get("info") or {}).get("gameType") == CUSTOM_GAME and len((match.get("info") or {}).get("participants") or []) == PLAYERS
            if usable and not self.budget(1):
                raise Unavailable()
            seats = seats_at(match, self.riot.timeline(match_id), self.minute) if usable else []
        except LinkError as exc:
            raise Unavailable() from exc
        self.table.put(MATCHES, match_id, {"seats": json.dumps(seats), "expires": int(self.clock()) + MATCH_SECONDS})
        return seats
