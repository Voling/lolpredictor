CREATE TABLE IF NOT EXISTS players (
    puuid           TEXT PRIMARY KEY,
    game_name       TEXT,
    tag_line        TEXT,
    riot_id         TEXT GENERATED ALWAYS AS (game_name || '#' || tag_line) STORED,
    tier            TEXT,
    division        TEXT,
    league_points   INTEGER,
    lp_value        INTEGER,
    wins            INTEGER,
    losses          INTEGER,
    depth           INTEGER,
    in_scope        BOOLEAN DEFAULT TRUE,
    discovered_at   TIMESTAMPTZ DEFAULT now(),
    crawled_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS players_riot_id ON players (riot_id);
CREATE INDEX IF NOT EXISTS players_name_lower ON players (lower(game_name), lower(tag_line));

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    platform        TEXT,
    queue_id        INTEGER,
    game_creation   TIMESTAMPTZ,
    game_duration   INTEGER,
    patch           TEXT,
    window_minutes  SMALLINT
);
CREATE INDEX IF NOT EXISTS matches_created ON matches (game_creation DESC);

CREATE TABLE IF NOT EXISTS participations (
    match_id        TEXT NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
    puuid           TEXT NOT NULL REFERENCES players (puuid),
    participant_id  SMALLINT,
    team_id         SMALLINT,
    position        TEXT,
    champion_id     INTEGER,
    champion_name   TEXT,
    win             BOOLEAN,
    PRIMARY KEY (match_id, puuid)
);
CREATE INDEX IF NOT EXISTS participations_puuid ON participations (puuid);
CREATE INDEX IF NOT EXISTS participations_champion ON participations (champion_name, position);

CREATE TABLE IF NOT EXISTS frames (
    match_id        TEXT NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
    puuid           TEXT NOT NULL,
    minute          SMALLINT NOT NULL,
    x               INTEGER,
    y               INTEGER,
    total_gold      INTEGER,
    current_gold    INTEGER,
    xp              INTEGER,
    minions         INTEGER,
    jungle_minions  INTEGER,
    level           SMALLINT,
    damage_done     INTEGER,
    damage_taken    INTEGER,
    health          INTEGER,
    health_max      INTEGER,
    movement_speed  INTEGER,
    PRIMARY KEY (match_id, puuid, minute)
);
CREATE INDEX IF NOT EXISTS frames_player_minute ON frames (puuid, minute);

CREATE TABLE IF NOT EXISTS events (
    match_id        TEXT NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
    event_index     INTEGER NOT NULL,
    minute          SMALLINT,
    timestamp_ms    INTEGER,
    type            TEXT,
    actor           TEXT,
    victim          TEXT,
    assists         TEXT[],
    x               INTEGER,
    y               INTEGER,
    ward_type       TEXT,
    monster_type    TEXT,
    building_type   TEXT,
    lane_type       TEXT,
    item_id         INTEGER,
    tower_type      TEXT,
    killer_team_id  SMALLINT,
    kill_type       TEXT,
    monster_sub_type TEXT,
    PRIMARY KEY (match_id, event_index)
);
CREATE INDEX IF NOT EXISTS events_type ON events (type, minute);
CREATE INDEX IF NOT EXISTS events_actor ON events (actor);

CREATE TABLE IF NOT EXISTS frontier (
    puuid       TEXT PRIMARY KEY,
    depth       INTEGER,
    priority    DOUBLE PRECISION,
    state       TEXT DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS frontier_state ON frontier (state, depth, priority DESC);
