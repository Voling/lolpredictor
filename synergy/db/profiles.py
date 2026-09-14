from ..config import Settings, get_settings
from ..ingest.store import Store

TOP_CHAMPIONS = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_profiles (
    puuid           TEXT PRIMARY KEY REFERENCES players (puuid) ON DELETE CASCADE,
    riot_id         TEXT,
    aliases         TEXT[],
    games           INTEGER NOT NULL,
    wins            INTEGER NOT NULL,
    winrate         REAL,
    main_position   TEXT,
    position_share  REAL,
    positions       TEXT[],
    champion_pool   INTEGER,
    top_champions   TEXT[],
    first_seen      TIMESTAMPTZ,
    last_seen       TIMESTAMPTZ,
    active_days     INTEGER,
    patches         INTEGER,
    tier            TEXT,
    division        TEXT,
    lp_value        INTEGER,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS player_profiles_games ON player_profiles (games DESC);
CREATE INDEX IF NOT EXISTS player_profiles_rank ON player_profiles (lp_value DESC NULLS LAST);
"""

BUILD = """
INSERT INTO player_profiles (
    puuid, riot_id, aliases, games, wins, winrate, main_position, position_share,
    positions, champion_pool, top_champions, first_seen, last_seen, active_days,
    patches, tier, division, lp_value, updated_at
)
WITH played AS (
    SELECT c.puuid, c.position, c.champion_name, c.win, m.game_creation, m.patch
    FROM participations c JOIN matches m ON m.match_id = c.match_id
),
totals AS (
    SELECT puuid, COUNT(*) AS games, COUNT(*) FILTER (WHERE win) AS wins,
           MIN(game_creation) AS first_seen, MAX(game_creation) AS last_seen,
           COUNT(DISTINCT game_creation::date) AS active_days,
           COUNT(DISTINCT patch) AS patches,
           COUNT(DISTINCT champion_name) AS champion_pool
    FROM played GROUP BY puuid
),
spots AS (
    SELECT puuid, position, COUNT(*) AS n,
           ROW_NUMBER() OVER (PARTITION BY puuid ORDER BY COUNT(*) DESC, position) AS seat
    FROM played WHERE position IS NOT NULL AND position <> '' GROUP BY puuid, position
),
picks AS (
    SELECT puuid, champion_name,
           ROW_NUMBER() OVER (PARTITION BY puuid ORDER BY COUNT(*) DESC, champion_name) AS seat
    FROM played WHERE champion_name IS NOT NULL GROUP BY puuid, champion_name
)
SELECT
    t.puuid,
    p.riot_id,
    COALESCE(
        (SELECT array_agg(DISTINCT n.game_name || '#' || n.tag_line)
         FROM player_names n WHERE n.puuid = t.puuid),
        ARRAY[]::TEXT[]
    ),
    t.games,
    t.wins,
    t.wins::real / NULLIF(t.games, 0),
    (SELECT s.position FROM spots s WHERE s.puuid = t.puuid AND s.seat = 1),
    (SELECT s.n::real / t.games FROM spots s WHERE s.puuid = t.puuid AND s.seat = 1),
    COALESCE(
        (SELECT array_agg(s.position ORDER BY s.seat) FROM spots s WHERE s.puuid = t.puuid),
        ARRAY[]::TEXT[]
    ),
    t.champion_pool,
    COALESCE(
        (SELECT array_agg(k.champion_name ORDER BY k.seat)
         FROM picks k WHERE k.puuid = t.puuid AND k.seat <= %s),
        ARRAY[]::TEXT[]
    ),
    t.first_seen,
    t.last_seen,
    t.active_days,
    t.patches,
    p.tier,
    p.division,
    p.lp_value,
    now()
FROM totals t JOIN players p ON p.puuid = t.puuid
ON CONFLICT (puuid) DO UPDATE SET
    riot_id = EXCLUDED.riot_id,
    aliases = EXCLUDED.aliases,
    games = EXCLUDED.games,
    wins = EXCLUDED.wins,
    winrate = EXCLUDED.winrate,
    main_position = EXCLUDED.main_position,
    position_share = EXCLUDED.position_share,
    positions = EXCLUDED.positions,
    champion_pool = EXCLUDED.champion_pool,
    top_champions = EXCLUDED.top_champions,
    first_seen = EXCLUDED.first_seen,
    last_seen = EXCLUDED.last_seen,
    active_days = EXCLUDED.active_days,
    patches = EXCLUDED.patches,
    tier = EXCLUDED.tier,
    division = EXCLUDED.division,
    lp_value = EXCLUDED.lp_value,
    updated_at = now()
"""


def build_profiles(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    store = Store(settings)
    try:
        with store._tx() as cursor:
            cursor.execute(SCHEMA)
            cursor.execute(BUILD, (TOP_CHAMPIONS,))
            written = cursor.rowcount
        with store.conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS profiles,"
                " COUNT(*) FILTER (WHERE tier IS NOT NULL) AS ranked,"
                " COUNT(*) FILTER (WHERE games >= 5) AS playable,"
                " COUNT(*) FILTER (WHERE cardinality(aliases) > 1) AS renamed,"
                " ROUND(AVG(games), 1) AS mean_games,"
                " MAX(games) AS most_games"
                " FROM player_profiles"
            )
            summary = dict(cursor.fetchone())
    finally:
        store.close()
    summary["written"] = int(written)
    return summary
