CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable(
    'frames', by_range('game_creation', INTERVAL '7 days'),
    migrate_data => true, if_not_exists => true
);
SELECT create_hypertable(
    'events', by_range('game_creation', INTERVAL '7 days'),
    migrate_data => true, if_not_exists => true
);

ALTER TABLE frames SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'match_id',
    timescaledb.compress_orderby = 'puuid, minute'
);
ALTER TABLE events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'match_id',
    timescaledb.compress_orderby = 'event_index'
);

SELECT add_compression_policy('frames', INTERVAL '14 days', if_not_exists => true);
SELECT add_compression_policy('events', INTERVAL '14 days', if_not_exists => true);
