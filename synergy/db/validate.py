from ..config import Settings, get_settings
from ..ingest.store import Store

TEAM_SIZE = 5
ROSTER = TEAM_SIZE * 2
WINDOW_FRAMES = 16

CHECKS = {
    "matches outside the season": (
        "SELECT COUNT(*) AS n FROM matches WHERE split_part(patch,'.',1) <> %s", True
    ),
    "matches with no patch": ("SELECT COUNT(*) AS n FROM matches WHERE patch IS NULL OR patch = ''", False),
    "matches off the ranked queue": (
        "SELECT COUNT(*) AS n FROM matches WHERE queue_id IS DISTINCT FROM %s", True
    ),
    "matches without ten participations": (
        "SELECT COUNT(*) AS n FROM (SELECT match_id FROM participations"
        " GROUP BY match_id HAVING COUNT(*) <> %s) q", True
    ),
    "matches without two teams of five": (
        "SELECT COUNT(*) AS n FROM (SELECT match_id FROM participations"
        " GROUP BY match_id, team_id HAVING COUNT(*) <> %s) q", True
    ),
    "matches where both teams won or lost": (
        "SELECT COUNT(*) AS n FROM (SELECT match_id FROM participations"
        " GROUP BY match_id HAVING COUNT(DISTINCT win) <> 2) q", False
    ),
    "matches riot did not complete": (
        "SELECT COUNT(*) AS n FROM matches WHERE end_result IS DISTINCT FROM 'GameComplete'", False
    ),
    "participations with no match": (
        "SELECT COUNT(*) AS n FROM participations c"
        " LEFT JOIN matches m ON m.match_id = c.match_id WHERE m.match_id IS NULL", False
    ),
    "participations with no player": (
        "SELECT COUNT(*) AS n FROM participations c"
        " LEFT JOIN players p ON p.puuid = c.puuid WHERE p.puuid IS NULL", False
    ),
    "frames with no match": (
        "SELECT COUNT(*) AS n FROM frames f"
        " LEFT JOIN matches m ON m.match_id = f.match_id WHERE m.match_id IS NULL", False
    ),
    "events with no match": (
        "SELECT COUNT(*) AS n FROM events e"
        " LEFT JOIN matches m ON m.match_id = e.match_id WHERE m.match_id IS NULL", False
    ),
    "matches with no frames": (
        "SELECT COUNT(*) AS n FROM matches m"
        " WHERE NOT EXISTS (SELECT 1 FROM frames f WHERE f.match_id = m.match_id)", False
    ),
    "player games short of the window in a full length match": (
        "SELECT COUNT(*) AS n FROM (SELECT f.match_id, f.puuid FROM frames f"
        " JOIN matches m ON m.match_id = f.match_id WHERE f.minute < %s"
        " AND m.game_duration >= %s GROUP BY f.match_id, f.puuid HAVING COUNT(*) <> %s) q", True
    ),
    "frames with a negative minute": (
        "SELECT COUNT(*) AS n FROM frames WHERE minute < 0", False
    ),
    "frames with no position": ("SELECT COUNT(*) AS n FROM frames WHERE x IS NULL OR y IS NULL", False),
    "frames off the map": (
        "SELECT COUNT(*) AS n FROM frames WHERE x < -2000 OR y < -2000 OR x > 17000 OR y > 17000", False
    ),
    "events with a negative minute": (
        "SELECT COUNT(*) AS n FROM events WHERE minute < 0", False
    ),
    "matches awaiting the window stage": (
        "SELECT COUNT(*) AS n FROM matches WHERE window_minutes IS NULL", False
    ),
    "players with a name but no tag": (
        "SELECT COUNT(*) AS n FROM players WHERE game_name IS NOT NULL AND tag_line IS NULL", False
    ),
    "aliases with no player": (
        "SELECT COUNT(*) AS n FROM player_names n"
        " LEFT JOIN players p ON p.puuid = n.puuid WHERE p.puuid IS NULL", False
    ),
    "matches with a negative duration": (
        "SELECT COUNT(*) AS n FROM matches WHERE game_duration < 0", False
    ),
}

ARGUMENTS = {
    "matches outside the season": lambda s: (str(s.season),),
    "matches off the ranked queue": lambda s: (s.queue_id,),
    "matches without ten participations": lambda s: (ROSTER,),
    "matches without two teams of five": lambda s: (TEAM_SIZE,),
    "player games short of the window in a full length match": lambda s: (
        WINDOW_FRAMES, WINDOW_FRAMES * 60, WINDOW_FRAMES
    ),
}


def validate(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    store = Store(settings)
    report: dict[str, int] = {}
    try:
        with store.conn.cursor() as cursor:
            for label, (sql, parametrised) in CHECKS.items():
                cursor.execute(sql, ARGUMENTS[label](settings) if parametrised else None)
                report[label] = int(cursor.fetchone()["n"])
    finally:
        store.close()
    return report
