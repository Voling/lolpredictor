# TimescaleDB

Postgres 17 with the TimescaleDB extension. The two large tables are hypertables with
columnar compression; everything else is a plain relational table.

## Layout

| table | kind | rows | why |
|---|---|---|---|
| `matches` | relational | 43,835 | keeps `PRIMARY KEY (match_id)`, which a hypertable cannot |
| `players` | relational | 35,857 | dimension table |
| `participations` | relational | 428,160 | dimension table, joined per match |
| `frontier` | relational | ~34,000 | crawl state, updated constantly |
| `frames` | **hypertable** | 12,795,420 | per player per minute |
| `events` | **hypertable** | 48,863,986 | positioned game events |

Mixing the two is the intended pattern: TimescaleDB is a Postgres extension, so the fact
tables partition by time while the dimension tables keep ordinary constraints and join
normally. Making everything a hypertable would pay chunk overhead on 11 MB of metadata for
no benefit.

## Compression

| table | before | after | ratio |
|---|---|---|---|
| `frames` | 2,938 MB | 235 MB | 12.5x |
| `events` | 5,429 MB | 283 MB | 19.2x |

Chunks span 7 days of `game_creation`. Compression is segmented by `match_id`, so one
match's rows compress into one batch and a single-match read decompresses only that segment.
The policy has a 14 day lag, leaving recent chunks uncompressed so the crawler writes at full
speed; inserts into compressed chunks work but cost more.

## Trade-offs measured

| | plain | hypertable |
|---|---|---|
| single-match read, execution | 3.53 ms | 2.5 ms |
| single-match read, planning | 0.2 ms | **77 ms** |
| ingest per match | 50.7 ms | 53.7 ms |

Planning dominates for single-row lookups because the planner weighs every chunk. Read
batches of matches rather than one at a time, or constrain `game_creation` so chunk exclusion
prunes the search. Execution is faster compressed, since columnar scans read less.

## Primary keys

TimescaleDB requires the partition column in every unique index, so:

| table | before | after |
|---|---|---|
| `frames` | `(match_id, puuid, minute)` | `(match_id, puuid, minute, game_creation)` |
| `events` | `(match_id, event_index)` | `(match_id, event_index, game_creation)` |

`game_creation` is functionally determined by `match_id` through the foreign key, so no new
duplicate is possible, but the database no longer enforces that. `matches` is unaffected and
still guarantees one row per match.

## Setup

`synergy/db/schema.sql` creates the tables; `synergy/db/timescale.sql` converts and compresses
them. Both run on a fresh database from `Store.__init__`. Set `USE_TIMESCALE=0` to skip the
extension and run on plain Postgres — the code does not depend on chunks.

`python -m synergy reingest --workers 8` rewrites `frames` and `events` from the raw timelines
on disk. It is idempotent and takes about 26 minutes over 43,835 matches.

## Backups

    docker exec lolpredictor-postgres-1 pg_dump -U synergy -d synergy --format=custom --compress=6 > backups/synergy-$(date +%Y%m%d).dump

913 MB from a 17 GB database. Verify with `pg_restore --list`. Git Bash rewrites container
paths, so stream to stdout rather than passing `-f /tmp/...`.
