import json
from datetime import datetime, timezone

import numpy as np
import pytest

from synergy.config import Settings
from synergy.deep.stream import SPAN, parsed_inputs
from synergy.features.build import _extractors
from synergy.features.posterior import DEFAULT_TABLE, ROLES, SIGMA_FILE
from synergy.features.reaction import ZONE_COLUMNS
from synergy.features.timeline import ParsedTimeline
from synergy.features.wards import ward_rows
from synergy.ingest.store import Store
from tests.test_features import build_match, build_timeline

SIGMA = {role: list(DEFAULT_TABLE) for role in ROLES}
TIGHT = {role: [100.0, 120.0, 140.0, 160.0, 180.0] for role in ROLES}
LANE = (1500, 6000)
ENEMY_JUNGLE = (10100, 7100)
RIVER = (5000, 10000)


def _ward(minute: float, creator: int = 1) -> dict:
    return {"type": "WARD_PLACED", "timestamp": int(minute * 60000), "creatorId": creator, "wardType": "YELLOW_TRINKET"}


def _zones(row: dict) -> dict[str, float]:
    return {column: row[column] for column in ZONE_COLUMNS}


class _Cursor:
    def __init__(self, written: dict):
        self.written = written

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        return None

    def fetchone(self):
        return {"game_creation": datetime(2026, 1, 1, tzinfo=timezone.utc)}

    def executemany(self, query, rows):
        self.written[query.split()[2]] = list(rows)


class _Connection:
    def __init__(self):
        self.written = {}

    def cursor(self):
        return _Cursor(self.written)

    def commit(self):
        return None


def test_each_ward_spreads_one_unit_of_mass_over_the_zones():
    # given
    match, timeline = build_match(), build_timeline(events=[_ward(1.5), _ward(4.2, creator=2), _ward(7.75, creator=7), _ward(12.0, creator=10)])

    # when
    rows = ward_rows(match, timeline, sigma=SIGMA)

    # then
    assert len(rows) == 4 and all(not {"zone", "x", "y", "distance_to_next_gank"} & set(row) for row in rows)
    for row in rows:
        masses = np.array(list(_zones(row).values()))
        assert masses.sum() == pytest.approx(1.0, abs=1e-5) and masses.min() >= 0.0
        assert all(isinstance(row[column], float) for column in ("in_enemy_jungle", "in_own_jungle", "in_river", "warded_near_next_gank"))


def test_a_ward_between_two_snapshots_in_different_zones_splits_its_mass_by_time():
    # given
    match = build_match()
    timeline = build_timeline(blue_top_position=LANE, events=[_ward(2.25), _ward(2.75)])
    timeline["info"]["frames"][3]["participantFrames"]["1"]["position"] = {"x": ENEMY_JUNGLE[0], "y": ENEMY_JUNGLE[1]}

    # when
    early, late = (_zones(row) for row in ward_rows(match, timeline, sigma=TIGHT))

    # then
    assert early["in_lane_own_side"] == pytest.approx(0.75, abs=0.01) and early["in_enemy_jungle"] == pytest.approx(0.25, abs=0.01)
    assert late["in_lane_own_side"] == pytest.approx(0.25, abs=0.01) and late["in_enemy_jungle"] == pytest.approx(0.75, abs=0.01)


def test_a_kill_just_before_a_ward_pulls_its_mass_to_where_the_kill_happened():
    # given
    match = build_match()
    kill = {"type": "CHAMPION_KILL", "timestamp": int(2.4 * 60000), "killerId": 1, "victimId": 7, "assistingParticipantIds": [], "position": {"x": RIVER[0], "y": RIVER[1]}}
    plain = build_timeline(blue_top_position=LANE, events=[_ward(2.45)])
    anchored = build_timeline(blue_top_position=LANE, events=[kill, _ward(2.45)])

    # when
    before = _zones(ward_rows(match, plain, sigma=SIGMA)[0])
    after = _zones(ward_rows(match, anchored, sigma=SIGMA)[0])

    # then
    assert after["in_river"] > 0.5 > before["in_river"]
    assert after["in_lane_own_side"] < before["in_lane_own_side"]


def test_a_ward_is_near_the_next_gank_by_the_chance_its_placer_stood_within_the_gank_radius():
    # given
    match = build_match()
    far = {"type": "CHAMPION_KILL", "timestamp": int(3.5 * 60000), "killerId": 2, "victimId": 6, "assistingParticipantIds": [], "position": {"x": 13000, "y": 2000}}
    close = {"type": "CHAMPION_KILL", "timestamp": int(5.5 * 60000), "killerId": 2, "victimId": 8, "assistingParticipantIds": [], "position": {"x": LANE[0], "y": LANE[1] + 500}}
    timeline = build_timeline(blue_top_position=LANE, events=[_ward(2.5), far, _ward(4.5), close, _ward(6.5)])

    # when
    near = [row["warded_near_next_gank"] for row in ward_rows(match, timeline, sigma=TIGHT)]

    # then
    assert near[0] == pytest.approx(0.0, abs=1e-6)
    assert near[1] == pytest.approx(1.0, abs=1e-6)
    assert near[2] == 0.0


def test_the_timeline_adapter_lists_events_as_the_stored_rows_read_back():
    # given
    events = [
        {"type": "ITEM_PURCHASED", "timestamp": 61000, "participantId": 3, "itemId": 1055},
        {"type": "WARD_PLACED", "timestamp": 61000, "creatorId": 2, "wardType": "YELLOW_TRINKET"},
        {"type": "CHAMPION_KILL", "timestamp": 60500, "killerId": 1, "victimId": 7, "assistingParticipantIds": [2, 11], "position": {"x": 5000, "y": 10000}},
        {"type": "BUILDING_KILL", "timestamp": 200000, "killerId": 0, "assistingParticipantIds": [4], "position": {"x": 981, "y": 10441}},
        {"type": "WARD_KILL", "timestamp": 300000, "killerId": 6, "wardType": "SIGHT_WARD"},
        {"type": "LEVEL_UP", "timestamp": 400000, "participantId": 9, "level": 2},
        {"type": "CHAMPION_KILL", "timestamp": (SPAN + 2) * 60000, "killerId": 6, "victimId": 1, "position": {"x": 7000, "y": 7000}},
    ]
    match, timeline = build_match(), build_timeline(events=events)
    store = Store.__new__(Store)
    store.conn = _Connection()

    # when
    store.ingest_timeline("NA1_1", timeline)
    _, _, adapted = parsed_inputs(ParsedTimeline(match, timeline))

    # then
    stored = sorted(store.conn.written["events"], key=lambda row: (row[4], row[2]))
    assert adapted == [row[4:11] for row in stored if row[3] < SPAN]
    assert len(adapted) == 6 and adapted[0][1:5] == ("CHAMPION_KILL", "p0", "p6", ["p1"])


def test_the_ward_extractor_reads_the_position_spread_of_the_settings_it_runs_with(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    table = {role: [100.0, 200.0, 300.0, 400.0, 500.0] for role in ROLES}
    (settings.model_dir / SIGMA_FILE).write_text(json.dumps({"sigma": table}), encoding="utf-8")

    # when
    extract = _extractors(settings)["wards"]

    # then
    assert extract.keywords["sigma"] == table
