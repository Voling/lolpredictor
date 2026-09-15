import math
from itertools import combinations
from typing import Any

from .anchors import dead_at, death_windows, position_anchors
from .regions import MAP_SPAN

CLOSE_RANGE = 2500.0
VISION_WARDS = {"YELLOW_TRINKET", "SIGHT_WARD", "CONTROL_WARD", "BLUE_TRINKET"}
EARLY_MINUTES = 15


def _participant_puuids(timeline: dict, match: dict | None = None) -> dict[int, str]:
    info = timeline.get("info", {})
    listed = info.get("participants") or []
    mapping = {int(p["participantId"]): p["puuid"] for p in listed if p.get("puuid")}
    if mapping:
        return mapping
    puuids = timeline.get("metadata", {}).get("participants") or []
    if not puuids and match is not None:
        puuids = match.get("metadata", {}).get("participants") or []
    return {index + 1: puuid for index, puuid in enumerate(puuids)}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


class ParsedTimeline:
    def __init__(self, match: dict, timeline: dict):
        self.match_id = timeline.get("metadata", {}).get("matchId") or match["metadata"]["matchId"]
        self.pid_to_puuid = _participant_puuids(timeline, match)
        self._anchors: dict | None = None
        self._deaths: dict | None = None
        self.puuid_to_pid = {puuid: pid for pid, puuid in self.pid_to_puuid.items()}
        self.roles: dict[int, str] = {}
        self.teams: dict[int, int] = {}
        for participant in match["info"]["participants"]:
            pid = int(participant.get("participantId", 0))
            if pid == 0:
                pid = self.puuid_to_pid.get(participant["puuid"], 0)
            role = (participant.get("teamPosition") or participant.get("individualPosition") or "").upper()
            self.roles[pid] = role
            self.teams[pid] = int(participant.get("teamId", 0))
        self.positions: dict[int, list[tuple[float, float]]] = {pid: [] for pid in self.pid_to_puuid}
        self.gold: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.xp: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.cs: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.jungle_cs: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.unspent_gold: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.damage_done: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.damage_taken: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.levels: dict[int, list[float]] = {pid: [] for pid in self.pid_to_puuid}
        self.minutes: list[float] = []
        self.events: list[dict] = []
        self._load(timeline)

    def _load(self, timeline: dict) -> None:
        frames = timeline.get("info", {}).get("frames") or []
        for frame in frames:
            minute = float(frame.get("timestamp", 0)) / 60000.0
            self.minutes.append(minute)
            participant_frames: dict[str, Any] = frame.get("participantFrames") or {}
            for key, payload in participant_frames.items():
                pid = int(payload.get("participantId", key))
                if pid not in self.positions:
                    continue
                position = payload.get("position") or {}
                self.positions[pid].append((float(position.get("x", 0.0)), float(position.get("y", 0.0))))
                self.gold[pid].append(float(payload.get("totalGold", 0.0)))
                self.xp[pid].append(float(payload.get("xp", 0.0)))
                self.cs[pid].append(
                    float(payload.get("minionsKilled", 0.0)) + float(payload.get("jungleMinionsKilled", 0.0))
                )
                self.jungle_cs[pid].append(float(payload.get("jungleMinionsKilled", 0.0)))
                self.unspent_gold[pid].append(float(payload.get("currentGold", 0.0)))
                damage = payload.get("damageStats") or {}
                self.damage_done[pid].append(float(damage.get("totalDamageDoneToChampions", 0.0)))
                self.damage_taken[pid].append(float(damage.get("totalDamageTaken", 0.0)))
                self.levels[pid].append(float(payload.get("level", 1.0)))
            for event in frame.get("events") or []:
                self.events.append(event)

    def at_minute(self, series: dict[int, list[float]], pid: int, minute: int) -> float:
        values = series.get(pid) or []
        if not values:
            return 0.0
        index = min(minute, len(values) - 1)
        return values[index]

    def role_pid(self, team: int, role: str) -> int | None:
        for pid, value in self.roles.items():
            if value == role and self.teams.get(pid) == team:
                return pid
        return None

    def kill_events(self) -> list[dict]:
        return [event for event in self.events if event.get("type") == "CHAMPION_KILL"]

    def involved(self, event: dict) -> set[int]:
        ids = {int(event.get("killerId", 0))}
        ids.update(int(pid) for pid in event.get("assistingParticipantIds") or [])
        ids.discard(0)
        return ids


    @property
    def deaths(self) -> dict[int, list[tuple[float, float]]]:
        if self._deaths is None:
            self._deaths = death_windows(self, float(len(self.minutes)))
        return self._deaths

    def alive_at(self, pid: int, minute: float) -> bool:
        return not dead_at(self.deaths.get(pid, ()), minute)

    @property
    def anchors(self) -> dict[int, list[tuple[float, float, float, int]]]:
        if self._anchors is None:
            self._anchors = position_anchors(self, float(len(self.minutes)))
        return self._anchors

    def position_at(
        self, pid: int, minute: float, ignore: int | None = None
    ) -> tuple[float, float] | None:
        track = self.positions.get(pid) or []
        if not track:
            return None
        index = min(max(int(minute + 0.5), 0), len(track) - 1)
        nearest = track[index]
        best = abs(minute - index)
        for when, x, y, source in self.anchors.get(pid, ()):
            if source == ignore:
                continue
            gap = abs(minute - when)
            if gap < best:
                best, nearest = gap, (x, y)
        return nearest


def pair_rows(match: dict, timeline: dict, span: int = EARLY_MINUTES) -> list[dict]:
    parsed = ParsedTimeline(match, timeline)
    kills = [
        event
        for event in parsed.kill_events()
        if float(event.get('timestamp', 0)) / 60000.0 <= span
    ]
    rows = []
    for team in (100, 200):
        members = [pid for pid, value in parsed.teams.items() if value == team and pid in parsed.pid_to_puuid]
        for left, right in combinations(sorted(members), 2):
            positions_a = (parsed.positions.get(left) or [])[: span + 1]
            positions_b = (parsed.positions.get(right) or [])[: span + 1]
            frames = min(len(positions_a), len(positions_b))
            distances = [_distance(positions_a[i], positions_b[i]) for i in range(frames)]
            lane_distances = distances[3:16]
            takedowns_a = set()
            takedowns_b = set()
            shared = 0
            deaths_a: list[float] = []
            deaths_b: list[float] = []
            for index, event in enumerate(kills):
                participants = parsed.involved(event)
                minute = float(event.get("timestamp", 0)) / 60000.0
                if left in participants:
                    takedowns_a.add(index)
                if right in participants:
                    takedowns_b.add(index)
                if left in participants and right in participants:
                    shared += 1
                if int(event.get("victimId", 0)) == left:
                    deaths_a.append(minute)
                if int(event.get("victimId", 0)) == right:
                    deaths_b.append(minute)
            union = len(takedowns_a | takedowns_b)
            co_deaths = sum(
                1 for a in deaths_a if any(abs(a - b) <= 0.25 for b in deaths_b)
            )
            gold_a = parsed.gold.get(left) or []
            gold_b = parsed.gold.get(right) or []
            deltas_a = [gold_a[i + 1] - gold_a[i] for i in range(len(gold_a) - 1)]
            deltas_b = [gold_b[i + 1] - gold_b[i] for i in range(len(gold_b) - 1)]
            rows.append(
                {
                    "match_id": parsed.match_id,
                    "puuid_a": parsed.pid_to_puuid[left],
                    "puuid_b": parsed.pid_to_puuid[right],
                    "team_id": team,
                    "win": int(bool(next(
                        p.get("win")
                        for p in match["info"]["participants"]
                        if int(p.get("teamId", 0)) == team
                    ))),
                    "pair_mean_distance": (sum(distances) / len(distances) / MAP_SPAN) if distances else 0.5,
                    "pair_lane_distance": (
                        sum(lane_distances) / len(lane_distances) / MAP_SPAN if lane_distances else 0.5
                    ),
                    "pair_close_share": (
                        sum(1 for d in distances if d < CLOSE_RANGE) / len(distances) if distances else 0.0
                    ),
                    "pair_co_takedown_rate": shared / union if union else 0.0,
                    "pair_co_death_rate": co_deaths / max(min(len(deaths_a), len(deaths_b)), 1),
                    "pair_gold_corr": _pearson(deltas_a, deltas_b),
                }
            )
    return rows


PAIR_TIMELINE_COLUMNS = [
    "pair_mean_distance",
    "pair_lane_distance",
    "pair_close_share",
    "pair_co_takedown_rate",
    "pair_co_death_rate",
    "pair_gold_corr",
]
