import numpy as np
import pandas as pd

from synergy.ml.interaction import TEAM_PAIRS, TEAM_SIZE, _crossed, _selves, pair_between, sides


def test_cross_term_equals_the_explicit_sum_over_pairs():
    # given
    rng = np.random.default_rng(0)
    team = rng.normal(size=(3, TEAM_SIZE, 4)).astype(np.float32)
    raw = rng.normal(size=(4, 4))
    matrix = 0.5 * (raw + raw.T)

    # when
    closed = np.asarray(_crossed(team, matrix, 4))

    # then
    explicit = np.zeros(3)
    for row in range(3):
        for a in range(TEAM_SIZE):
            for b in range(a + 1, TEAM_SIZE):
                explicit[row] += team[row, a] @ matrix @ team[row, b]
    assert np.allclose(closed, explicit / (TEAM_PAIRS * 4), atol=1e-5)


def test_self_term_sums_each_seat_alone():
    # given
    team = np.ones((2, TEAM_SIZE, 3), np.float32)
    matrix = np.eye(3)

    # when
    value = np.asarray(_selves(team, matrix, 3))

    # then
    assert np.allclose(value, TEAM_SIZE * 3 / (TEAM_SIZE * 3))


def test_sides_splits_blue_and_red_in_seat_order():
    # given
    reduced = np.arange(20, dtype=np.float32).reshape(1, 10, 2)
    seat_side = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], np.int8)

    # when
    blue, red = sides(seat_side, reduced)

    # then
    assert blue[0, :, 0].tolist() == [0, 4, 8, 12, 16]
    assert red[0, :, 0].tolist() == [2, 6, 10, 14, 18]


def test_pair_between_reads_the_saved_matrix_and_ranks_against_the_corpus(tmp_path):
    # given
    from synergy.config import Settings

    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    matrix = np.eye(2)
    np.savez(settings.model_dir / "interaction_matrix.npz", matrix=matrix)
    pd.DataFrame(
        {"puuid": ["a", "b", "c"], "c0": [1.0, 1.0, -1.0], "c1": [0.0, 0.0, 0.0], "seats": [5, 7, 2]}
    ).to_parquet(settings.processed_dir / "player_styles.parquet", index=False)
    np.savez(
        settings.model_dir / "interaction_scores.npz",
        quantiles=np.linspace(-1.0, 1.0, 1001) / (TEAM_PAIRS * 2),
    )

    # when
    alike = pair_between("a", "b", settings)
    opposed = pair_between("a", "c", settings)
    unknown = pair_between("a", "zzz", settings)

    # then
    assert alike["synergy"] > 0 > opposed["synergy"]
    assert alike["percentile"] > 50 > opposed["percentile"]
    assert alike["left_seats"] == 5 and alike["right_seats"] == 7
    assert unknown is None
