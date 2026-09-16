import numpy as np
import pandas as pd
import pytest

from synergy.ml.interaction import (
    TEAM_PAIRS,
    TEAM_SIZE,
    NoGamesInPosition,
    _crossed,
    _dense,
    _selves,
    fully_seated,
    lineup_between,
    pair_between,
    position_profile,
    sides,
    standardise,
)


def test_cross_term_equals_the_explicit_sum_over_pairs_through_the_dense_matrix():
    # given
    rng = np.random.default_rng(0)
    team = rng.normal(size=(3, TEAM_SIZE, 4)).astype(np.float32)
    loading = rng.normal(size=(2, 4)).astype(np.float32)
    weight = rng.normal(size=2).astype(np.float32)
    matrix = np.asarray(_dense({"c_loading": loading, "c_weight": weight}, "c"))

    # when
    closed = np.asarray(_crossed(team, (loading, weight), 4))

    # then
    explicit = np.zeros(3)
    for row in range(3):
        for a in range(TEAM_SIZE):
            for b in range(a + 1, TEAM_SIZE):
                explicit[row] += team[row, a] @ matrix @ team[row, b]
    assert np.allclose(closed, explicit / (TEAM_PAIRS * 4), atol=1e-4)


def test_self_term_sums_each_seat_alone():
    # given
    team = np.ones((2, TEAM_SIZE, 3), np.float32)
    identity = (np.eye(3, dtype=np.float32), np.ones(3, np.float32))

    # when
    value = np.asarray(_selves(team, identity, 3))

    # then
    assert np.allclose(value, TEAM_SIZE * 3 / (TEAM_SIZE * 3))


def test_a_factored_matrix_is_symmetric_and_low_rank():
    # given
    rng = np.random.default_rng(1)
    drawn = {"x_loading": rng.normal(size=(3, 7)), "x_weight": rng.normal(size=3)}

    # when
    matrix = np.asarray(_dense(drawn, "x"))

    # then
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.matrix_rank(matrix) == 3


def test_standardise_uses_only_the_fit_rows_and_zeroes_the_unknown():
    # given
    encoding = np.array([[[1.0, np.nan]], [[3.0, 4.0]], [[100.0, 0.0]]])

    # when
    reduced, centre, spread = standardise(encoding, np.array([0, 1]))

    # then
    assert np.allclose(centre, [2.0, 4.0])
    assert reduced[0, 0, 1] == 0.0
    assert reduced[2, 0, 0] > 10.0


def test_sides_splits_blue_and_red_in_seat_order():
    # given
    reduced = np.arange(20, dtype=np.float32).reshape(1, 10, 2)
    seat_side = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], np.int8)

    # when
    blue, red = sides(seat_side, reduced)

    # then
    assert blue[0, :, 0].tolist() == [0, 4, 8, 12, 16]
    assert red[0, :, 0].tolist() == [2, 6, 10, 14, 18]


def test_a_match_with_any_seat_missing_a_position_is_dropped_from_the_fit():
    # given
    positions = np.array(
        [
            ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
            ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UNKNOWN", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
            ["UNKNOWN"] * 10,
        ]
    )

    # when
    keep = fully_seated(positions)

    # then
    assert keep.tolist() == [True, False, False]


def _fitted(tmp_path):
    from synergy.config import Settings

    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    np.savez(
        settings.model_dir / "interaction_matrix.npz",
        matrix=np.array([[1.0, 0.0], [0.0, -1.0]]),
        columns=np.array(["tend_dive_high_own", "rsp_kill_ours_near_converged"]),
        centre=np.zeros(2), spread=np.ones(2),
    )
    pd.DataFrame(
        {
            "puuid": ["a", "a", "b", "c", "d", "e"],
            "position": ["TOP", "JUNGLE", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
            "tend_dive_high_own": [1.0, 0.0, 1.0, 0.0, 1.0, 0.5],
            "rsp_kill_ours_near_converged": [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
            "seats": [5, 1, 7, 2, 3, 4],
            "evidence": [0.8, 0.1, 0.9, 0.4, 0.6, 0.7],
        }
    ).to_parquet(settings.processed_dir / "player_styles.parquet", index=False)
    grid = np.linspace(-1.0, 1.0, 1001) / (TEAM_PAIRS * 2)
    np.savez(
        settings.model_dir / "interaction_scores.npz",
        quantiles=grid, combos=np.array(["JUNGLE+TOP", "MIDDLE+TOP"]),
        combo_quantiles=np.stack([grid, grid * 4.0]), team_quantiles=grid * 10.0,
    )
    return settings


def test_pair_between_reads_each_player_in_the_given_position_and_ranks_within_the_position_pair(tmp_path):
    # given
    settings = _fitted(tmp_path)

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    unrelated = pair_between("a", "TOP", "c", "MIDDLE", settings)
    off_role = pair_between("a", "JUNGLE", "c", "MIDDLE", settings)

    # then
    assert alike["synergy"] > 0 and alike["percentile"] > 50
    assert alike["drivers"][0]["left"] == "tend_dive_high_own" and alike["drivers"][0]["right"] == "tend_dive_high_own"
    assert alike["left_games"] == 5 and alike["right_games"] == 7
    assert alike["positions"] == {"left": "TOP", "right": "JUNGLE"}
    assert unrelated["synergy"] == 0.0
    assert off_role["synergy"] < 0 and off_role["left_games"] == 1
    assert alike["reliable"] is False and "inspection only" in alike["note"]
    reading = alike["reading"]["left"]
    assert reading["distinctive"][0]["cell"] == "tend_dive_high_own" and "dives" in reading["distinctive"][0]["words"]
    assert reading["distinctive"][0]["percentile"] == 0.0
    assert np.isclose(sum(item["contribution"] for item in reading["situations"]), alike["synergy"], atol=1e-5)
    assert reading["situations"][0]["words"] == "diving"
    (settings.model_dir / "interaction_report.json").write_text('{"informative": true}', encoding="utf-8")
    assert pair_between("a", "TOP", "b", "JUNGLE", settings)["reliable"] is True


def test_pair_between_refuses_a_shared_position_and_a_position_never_played(tmp_path):
    # given
    from synergy.config import Settings

    settings = _fitted(tmp_path)

    # when
    with pytest.raises(ValueError) as shared:
        pair_between("a", "TOP", "b", "TOP", settings)
    with pytest.raises(NoGamesInPosition) as never:
        pair_between("a", "TOP", "b", "UTILITY", settings)

    # then
    assert "two different positions" in str(shared.value)
    assert never.value.position == "UTILITY" and "no games as UTILITY" in str(never.value)
    assert position_profile("b", "UTILITY", settings) == {"games": 0, "evidence": 0.0}
    assert position_profile("b", "JUNGLE", settings) == {"games": 7, "evidence": 0.9}
    assert position_profile("b", "JUNGLE", Settings(data_dir=tmp_path / "empty")) is None
    assert pair_between("a", "TOP", "b", "JUNGLE", settings)["right_evidence"] == 0.9


def test_a_lineup_sums_its_ten_pairs_and_ranks_against_corpus_teams(tmp_path):
    # given
    settings = _fitted(tmp_path)
    five = {"TOP": "a", "JUNGLE": "b", "MIDDLE": "c", "BOTTOM": "d", "UTILITY": "e"}

    # when
    result = lineup_between(five, settings)
    with pytest.raises(ValueError):
        lineup_between({**five, "UTILITY": "a"}, settings)
    with pytest.raises(ValueError):
        lineup_between({key: value for key, value in five.items() if key != "UTILITY"}, settings)

    # then
    assert len(result["pairs"]) == 10
    assert np.isclose(result["synergy"], sum(pair["synergy"] for pair in result["pairs"]), atol=1e-5)
    assert 0.0 <= result["percentile"] <= 100.0
    assert result["pairs"][0]["synergy"] >= result["pairs"][-1]["synergy"]
