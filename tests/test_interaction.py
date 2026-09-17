import numpy as np

from synergy.ml.interaction import (
    TEAM_PAIRS,
    TEAM_SIZE,
    _crossed,
    _dense,
    _selves,
    fully_seated,
    shuffle_seats,
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


def test_shuffling_seats_into_a_buffer_draws_the_same_permutations_as_stacking():
    # given
    team = np.random.default_rng(3).normal(size=(9, TEAM_SIZE, 4)).astype(np.float32)
    stacked_rng, buffered_rng = np.random.default_rng(7), np.random.default_rng(7)

    # when
    stacked = np.stack([team[stacked_rng.permutation(len(team)), seat] for seat in range(TEAM_SIZE)], axis=1)
    buffered = shuffle_seats(team, buffered_rng, np.empty_like(team))

    # then
    assert np.array_equal(stacked, buffered)
    assert stacked_rng.random() == buffered_rng.random()


def test_standardise_leaves_its_input_untouched():
    # given
    encoding = np.array([[[1.0, np.nan]], [[3.0, 4.0]]], dtype=np.float32)
    before = encoding.copy()

    # when
    reduced, _, _ = standardise(encoding, np.array([0, 1]))

    # then
    assert reduced.dtype == np.float32
    assert np.array_equal(encoding, before, equal_nan=True)
    assert not np.shares_memory(encoding, reduced)


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
