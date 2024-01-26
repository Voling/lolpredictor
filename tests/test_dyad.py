import pytest

from synergy.features.dyad import DYAD_FEATURE_COLUMNS, dyad_rows
from synergy.features.extra import EXTRA_FEATURE_COLUMNS, extra_rows
from tests.test_features import build_match, build_timeline

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _rows(**kwargs):
    return dyad_rows(build_match(), build_timeline(**kwargs))


def test_every_same_team_pair_appears_once():
    rows = _rows()
    assert len(rows) == 20
    keys = {(row["team_id"], row["puuid_a"], row["puuid_b"]) for row in rows}
    assert len(keys) == 20


def test_pairs_never_cross_teams():
    match = build_match()
    teams = {p["puuid"]: p["teamId"] for p in match["info"]["participants"]}
    for row in dyad_rows(match, build_timeline()):
        assert teams[row["puuid_a"]] == teams[row["puuid_b"]]


def test_every_declared_column_is_produced():
    row = _rows()[0]
    assert set(DYAD_FEATURE_COLUMNS) <= set(row)


def test_distance_ignores_the_shared_spawn():
    apart = _rows(blue_top_position=(1000, 14000))
    row = next(r for r in apart if r["role_a"] == "TOP" or r["role_b"] == "TOP")
    assert row["d_min_distance"] > 0.0


def test_close_share_bands_are_monotone():
    for row in _rows():
        assert row["d_close_share_1500"] <= row["d_close_share_3000"] <= row["d_close_share_5000"]


def test_rates_stay_within_bounds():
    for row in _rows():
        for column in ("d_co_takedown_rate", "d_co_death_rate", "d_avenged_rate", "d_zone_overlap"):
            assert 0.0 <= row[column] <= 1.0
        assert -1.0 <= row["d_lead_lag"] <= 1.0
        assert -1.0 <= row["d_gold_corr"] <= 1.0


def test_a_short_game_produces_nothing():
    assert dyad_rows(build_match(), build_timeline(minutes=4)) == []


def test_extra_rows_cover_every_participant():
    rows = extra_rows(build_match(), build_timeline())
    assert len(rows) == 10
    assert set(EXTRA_FEATURE_COLUMNS) <= set(rows[0])


def test_extra_shares_are_fractions():
    for row in extra_rows(build_match(), build_timeline()):
        for column in ("x_physical_share", "x_true_share", "x_taken_physical_share", "x_wasted_damage"):
            assert 0.0 <= row[column] <= 1.0


def test_first_blood_is_flagged_for_one_player_at_most():
    rows = extra_rows(build_match(), build_timeline())
    assert sum(row["x_took_first_blood"] for row in rows) <= 1
