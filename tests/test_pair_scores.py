import numpy as np

from synergy.ml.interaction import TEAM_PAIRS, _side_pairs


def test_each_team_pair_scores_its_two_styles_through_the_matrix():
    # given
    rng = np.random.default_rng(4)
    reduced = rng.normal(size=(7, 10, 6)).astype(np.float32)
    matrix = rng.normal(size=(6, 6))
    side = np.array([[1, 0] * 5] * 7)
    basis = {
        "reduced": reduced,
        "seat_side": side,
        "seat_puuid": np.array([[f"m{m}s{s}" for s in range(10)] for m in range(7)], dtype=object),
        "seat_position": np.array([["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2 for _ in range(7)], dtype=object),
        "match_id": np.array([f"m{m}" for m in range(7)], dtype=object),
    }

    # when
    pairs = _side_pairs(basis, matrix, 1)

    # then
    first = pairs.iloc[0]
    left, right = reduced[0, 0].astype(float), reduced[0, 2].astype(float)
    assert len(pairs) == 7 * 10
    assert first["puuid_a"] == "m0s0" and first["puuid_b"] == "m0s2"
    assert np.isclose(first["synergy"], left @ matrix @ right / (TEAM_PAIRS * 6), rtol=1e-12)
