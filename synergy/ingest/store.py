import gzip
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
TIMESCALE = Path(__file__).resolve().parent.parent / "db" / "timescale.sql"
ACTOR_KEYS = ("killerId", "creatorId", "participantId")
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


COMPRESS_LEVEL = 6


def _write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=COMPRESS_LEVEL) as handle:
        json.dump(payload, handle, separators=(",", ":"))
    tmp.replace(path)


def _read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _moment(milliseconds: Any) -> datetime | None:
    if not milliseconds:
        return None
    return datetime.fromtimestamp(int(milliseconds) / 1000, timezone.utc)


def _position(participant: dict) -> str:
    raw = (participant.get("teamPosition") or participant.get("individualPosition") or "").upper()
    return raw if raw in POSITIONS else "UNKNOWN"


class Store:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.conn = psycopg.connect(self.settings.database_url, row_factory=dict_row, autocommit=False)
        if not self._schema_ready():
            with self._tx() as cursor:
                cursor.execute(SCHEMA.read_text(encoding="utf-8"))
            self._apply_timescale()

    def _apply_timescale(self) -> None:
        if not self.settings.timescale:
            return
        try:
            with self._tx() as cursor:
                cursor.execute(TIMESCALE.read_text(encoding="utf-8"))
        except psycopg.Error as exc:
            logger.info("timescale setup skipped: %s", exc)

    def _schema_ready(self) -> bool:
        with self._tx() as cursor:
            cursor.execute("SELECT to_regclass('matches') AS present")
            return cursor.fetchone()["present"] is not None

    @contextmanager
    def _tx(self):
        try:
            with self.conn.cursor() as cursor:
                yield cursor
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def match_path(self, match_id: str) -> Path:
        return self.settings.match_dir / f"{match_id}.json.gz"

    def timeline_path(self, match_id: str) -> Path:
        return self.settings.timeline_dir / f"{match_id}.json.gz"

    def window_path(self, match_id: str) -> Path:
        return self.settings.window_dir / f"{match_id}.json.gz"

    def has_match(self, match_id: str) -> bool:
        return self.match_path(match_id).exists()

    def has_timeline(self, match_id: str) -> bool:
        return self.timeline_path(match_id).exists()

    def has_window(self, match_id: str) -> bool:
        return self.window_path(match_id).exists()

    def load_match(self, match_id: str) -> dict:
        return _read_json_gz(self.match_path(match_id))

    def load_timeline(self, match_id: str) -> dict:
        return _read_json_gz(self.timeline_path(match_id))

    def load_window(self, match_id: str) -> dict:
        return _read_json_gz(self.window_path(match_id))

    def save_window(self, match_id: str, timeline: dict) -> None:
        _write_json_gz(self.window_path(match_id), timeline)
        with self._tx() as cursor:
            cursor.execute(
                "UPDATE matches SET window_minutes=%s WHERE match_id=%s",
                (timeline.get("info", {}).get("windowMinutes"), match_id),
            )

    def save_match(self, match: dict) -> None:
        info = match["info"]
        match_id = match["metadata"]["matchId"]
        _write_json_gz(self.match_path(match_id), match)
        patch = ".".join(str(info.get("gameVersion", "")).split(".")[:2])
        with self._tx() as cursor:
            cursor.execute(
                "INSERT INTO matches (match_id, platform, queue_id, game_creation, game_duration, patch)"
                " VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (match_id) DO UPDATE SET"
                " platform=EXCLUDED.platform, queue_id=EXCLUDED.queue_id,"
                " game_creation=EXCLUDED.game_creation, game_duration=EXCLUDED.game_duration,"
                " patch=EXCLUDED.patch",
                (
                    match_id,
                    info.get("platformId"),
                    info.get("queueId"),
                    _moment(info.get("gameCreation")),
                    info.get("gameDuration"),
                    patch,
                ),
            )
            self._record_identities(cursor, info["participants"])
            cursor.execute("DELETE FROM participations WHERE match_id=%s", (match_id,))
            cursor.executemany(
                "INSERT INTO participations (match_id, puuid, participant_id, team_id, position,"
                " champion_id, champion_name, win) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        match_id,
                        p["puuid"],
                        p.get("participantId"),
                        p.get("teamId"),
                        _position(p),
                        p.get("championId"),
                        p.get("championName"),
                        bool(p.get("win")),
                    )
                    for p in info["participants"]
                ],
            )

    def save_timeline(self, match_id: str, timeline: dict) -> None:
        _write_json_gz(self.timeline_path(match_id), timeline)
        self.ingest_timeline(match_id, timeline)

    def ingest_timeline(self, match_id: str, timeline: dict) -> tuple[int, int]:
        info = timeline.get("info", {})
        puuids = {
            int(entry["participantId"]): entry["puuid"]
            for entry in info.get("participants") or []
            if entry.get("puuid")
        }
        if not puuids:
            listed = timeline.get("metadata", {}).get("participants") or []
            puuids = {index + 1: puuid for index, puuid in enumerate(listed)}
        frames, events = [], []
        for minute, frame in enumerate(info.get("frames") or []):
            for key, payload in (frame.get("participantFrames") or {}).items():
                pid = int(payload.get("participantId", key))
                puuid = puuids.get(pid)
                if puuid is None:
                    continue
                position = payload.get("position") or {}
                damage = payload.get("damageStats") or {}
                stats = payload.get("championStats") or {}
                frames.append(
                    (
                        match_id,
                        puuid,
                        minute,
                        position.get("x"),
                        position.get("y"),
                        payload.get("totalGold"),
                        payload.get("currentGold"),
                        payload.get("xp"),
                        payload.get("minionsKilled"),
                        payload.get("jungleMinionsKilled"),
                        payload.get("level"),
                        damage.get("totalDamageDoneToChampions"),
                        damage.get("totalDamageTaken"),
                        stats.get("health"),
                        stats.get("healthMax"),
                        stats.get("movementSpeed"),
                    )
                )
            for event in frame.get("events") or []:
                index = len(events)
                position = event.get("position") or {}
                actor = next(
                    (int(event[key]) for key in ACTOR_KEYS if event.get(key)), None
                )
                events.append(
                    (
                        match_id,
                        index,
                        int(event.get("timestamp", 0)) // 60000,
                        event.get("timestamp"),
                        event.get("type"),
                        puuids.get(actor),
                        puuids.get(int(event["victimId"])) if event.get("victimId") else None,
                        [
                            puuids[int(pid)]
                            for pid in event.get("assistingParticipantIds") or []
                            if int(pid) in puuids
                        ],
                        position.get("x"),
                        position.get("y"),
                        event.get("wardType"),
                        event.get("monsterType"),
                        event.get("buildingType"),
                        event.get("laneType"),
                        event.get("itemId"),
                        event.get("towerType"),
                        event.get("killerTeamId"),
                        event.get("killType"),
                        event.get("monsterSubType"),
                    )
                )
        with self._tx() as cursor:
            cursor.execute("DELETE FROM frames WHERE match_id=%s", (match_id,))
            cursor.execute("DELETE FROM events WHERE match_id=%s", (match_id,))
            if frames:
                cursor.executemany(
                    "INSERT INTO frames (match_id, puuid, minute, x, y, total_gold, current_gold, xp,"
                    " minions, jungle_minions, level, damage_done, damage_taken,"
                    " health, health_max, movement_speed)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    frames,
                )
            if events:
                cursor.executemany(
                    "INSERT INTO events (match_id, event_index, minute, timestamp_ms, type, actor,"
                    " victim, assists, x, y, ward_type, monster_type, building_type, lane_type,"
                    " item_id, tower_type, killer_team_id, kill_type, monster_sub_type)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    events,
                )
        return len(frames), len(events)

    def match_ids(self) -> list[str]:
        with self._tx() as cursor:
            cursor.execute("SELECT match_id FROM matches ORDER BY game_creation NULLS LAST, match_id")
            return [row["match_id"] for row in cursor.fetchall()]

    def _record_identities(self, cursor, participants: list[dict]) -> int:
        cursor.executemany(
            "INSERT INTO players (puuid) VALUES (%s) ON CONFLICT (puuid) DO NOTHING",
            [(p["puuid"],) for p in participants],
        )
        known = [
            (p["riotIdGameName"], p["riotIdTagline"], p["puuid"])
            for p in participants
            if p.get("riotIdGameName") and p.get("riotIdTagline")
        ]
        if known:
            cursor.executemany(
                "UPDATE players SET game_name=%s, tag_line=%s WHERE puuid=%s AND game_name IS NULL",
                known,
            )
        return len(known)

    def record_identities(self, participants: list[dict]) -> int:
        with self._tx() as cursor:
            count = self._record_identities(cursor, participants)
        return count

    def backfill_identities(self) -> int:
        named = 0
        with self._tx() as cursor:
            for match_id in self.match_ids():
                try:
                    match = self.load_match(match_id)
                except FileNotFoundError:
                    continue
                named += self._record_identities(cursor, match["info"]["participants"])
        return named

    def upsert_player(self, puuid: str, **fields: Any) -> None:
        with self._tx() as cursor:
            cursor.execute(
                "INSERT INTO players (puuid) VALUES (%s) ON CONFLICT (puuid) DO NOTHING", (puuid,)
            )
            if fields:
                columns = ", ".join(f"{key}=%s" for key in fields)
                cursor.execute(
                    f"UPDATE players SET {columns} WHERE puuid=%s", (*fields.values(), puuid)
                )

    def get_player(self, puuid: str) -> dict | None:
        with self._tx() as cursor:
            cursor.execute("SELECT * FROM players WHERE puuid=%s", (puuid,))
            return cursor.fetchone()

    def find_player_by_riot_id(self, name: str, tag: str) -> dict | None:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM players WHERE lower(game_name)=lower(%s) AND lower(tag_line)=lower(%s)",
                (name, tag),
            )
            return cursor.fetchone()

    def players(self) -> list[dict]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM players")
            return cursor.fetchall()

    def match_rank_summary(self) -> list[dict]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT c.match_id, m.game_creation, COUNT(p.lp_value) AS ranked,"
                " AVG(p.lp_value) AS average_lp"
                " FROM participations c JOIN matches m ON m.match_id = c.match_id"
                " LEFT JOIN players p ON p.puuid = c.puuid"
                " GROUP BY c.match_id, m.game_creation"
            )
            return cursor.fetchall()

    def unranked_players(self, min_games: int = 1) -> list[str]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT p.puuid FROM players p JOIN participations c ON c.puuid = p.puuid"
                " WHERE p.tier IS NULL GROUP BY p.puuid HAVING COUNT(*) >= %s"
                " ORDER BY COUNT(*) DESC",
                (min_games,),
            )
            return [row["puuid"] for row in cursor.fetchall()]

    def unnamed_players(self, min_games: int) -> list[str]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT p.puuid FROM players p JOIN participations c ON c.puuid = p.puuid"
                " WHERE p.game_name IS NULL GROUP BY p.puuid HAVING COUNT(*) >= %s"
                " ORDER BY COUNT(*) DESC",
                (min_games,),
            )
            return [row["puuid"] for row in cursor.fetchall()]

    def push_frontier(self, puuid: str, depth: int, priority: float, requeue: bool = False) -> None:
        update = (
            " depth=LEAST(frontier.depth, EXCLUDED.depth),"
            " priority=GREATEST(frontier.priority, EXCLUDED.priority), state='pending'"
            if requeue
            else " depth=LEAST(frontier.depth, EXCLUDED.depth), priority=frontier.priority + 1"
        )
        with self._tx() as cursor:
            cursor.execute(
                "INSERT INTO frontier (puuid, depth, priority, state) VALUES (%s,%s,%s,'pending')"
                " ON CONFLICT (puuid) DO UPDATE SET" + update,
                (puuid, depth, priority),
            )

    def pop_frontier(self, limit: int = 1) -> list[dict]:
        with self._tx() as cursor:
            cursor.execute(
                "WITH picked AS ("
                " SELECT puuid FROM frontier WHERE state='pending'"
                " ORDER BY depth ASC, priority DESC LIMIT %s FOR UPDATE SKIP LOCKED),"
                " taken AS (UPDATE frontier SET state='active'"
                " WHERE puuid IN (SELECT puuid FROM picked)"
                " RETURNING puuid, depth, priority, state)"
                " SELECT * FROM taken ORDER BY depth ASC, priority DESC",
                (limit,),
            )
            rows = cursor.fetchall()
        return rows

    def mark_frontier(self, puuid: str, state: str) -> None:
        with self._tx() as cursor:
            cursor.execute("UPDATE frontier SET state=%s WHERE puuid=%s", (state, puuid))

    def reset_active(self) -> None:
        with self._tx() as cursor:
            cursor.execute("UPDATE frontier SET state='pending' WHERE state='active'")

    def frontier_counts(self) -> dict[str, int]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT state, COUNT(*) AS total FROM frontier GROUP BY state")
            return {row["state"]: row["total"] for row in cursor.fetchall()}

    def counts(self) -> dict[str, int]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT (SELECT COUNT(*) FROM matches) AS matches,"
                " (SELECT COUNT(DISTINCT match_id) FROM frames) AS timelines,"
                " (SELECT COUNT(*) FROM matches WHERE window_minutes IS NOT NULL) AS windows,"
                " (SELECT COUNT(*) FROM players) AS players,"
                " (SELECT COUNT(*) FROM participations) AS participations,"
                " (SELECT COUNT(*) FROM frames) AS frames,"
                " (SELECT COUNT(*) FROM events) AS events"
            )
            return dict(cursor.fetchone())
