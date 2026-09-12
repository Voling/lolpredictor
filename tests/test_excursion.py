from synergy.features.excursion import EXCURSION_COLUMNS, excursion_rows
from tests.test_features import build_match, build_timeline


def test_every_participant_gets_a_row_with_every_column():
    rows = excursion_rows(build_match(), build_timeline())
    assert len(rows) == 10
    for row in rows:
        for column in EXCURSION_COLUMNS:
            assert column in row
            assert row[column] == row[column]


def test_shares_stay_within_bounds():
    for row in excursion_rows(build_match(), build_timeline()):
        for column in ("x_hidden_share", "x_return_share", "x_position_doubt"):
            assert 0.0 <= row[column] <= 1.0


def test_a_short_game_produces_nothing():
    timeline = build_timeline(4)
    assert excursion_rows(build_match(), timeline) == []


def test_one_shop_trip_counts_once_however_many_items_are_bought():
    from synergy.features.excursion import VISIT_GAP

    assert VISIT_GAP > 0
    rows = {row["puuid"]: row for row in excursion_rows(build_match(), build_timeline())}
    for row in rows.values():
        assert row["x_base_visits"] <= row["x_anchors_pm"] * 15


def test_a_players_own_death_still_counts_as_a_position_they_occupied():
    from synergy.features.timeline import ParsedTimeline

    match, timeline = build_match(), build_timeline()
    parsed = ParsedTimeline(match, timeline)
    for pid, windows in parsed.deaths.items():
        for start, _ in windows:
            assert not parsed.alive_at(pid, start)
            assert any(abs(a[0] - start) < 1e-9 for a in parsed.anchors.get(pid, ()))
