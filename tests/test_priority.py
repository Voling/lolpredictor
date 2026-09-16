import numpy as np
import pandas as pd

from synergy.features.posterior import DEFAULT_TABLE, bridge_at, progress_moments
from synergy.features.priority import OUTCOMES, PRIORITY_COLUMNS, SITUATIONS, band_weights, lane_counts, lane_priority


def test_priority_goes_to_the_side_whose_lane_front_sits_on_the_enemy_half():
    # given
    ones = np.ones(2)
    front_deep = np.array([1.25, 0.75])

    # when
    blue_prio, blue_gap = lane_priority(front_deep, ones * 0.0, ones, front_deep, ones * 0.0, ones, blue=True)
    red_prio, red_gap = lane_priority(front_deep, ones * 0.0, ones, front_deep, ones * 0.0, ones, blue=False)

    # then
    assert blue_prio.tolist() == [1.0, 0.0] and red_prio.tolist() == [0.0, 1.0]
    assert np.allclose(blue_gap, [0.25, -0.25]) and np.allclose(red_gap, [-0.25, 0.25])


def test_an_empty_enemy_lane_hands_priority_to_whoever_is_there():
    # given
    u = np.array([0.9])
    nobody = np.array([0.0])
    somebody = np.array([1.0])

    # when
    held, _ = lane_priority(u, np.zeros(1), somebody, u, np.zeros(1), nobody, blue=True)
    absent, _ = lane_priority(u, np.zeros(1), nobody, u, np.zeros(1), somebody, blue=True)

    # then
    assert held.tolist() == [1.0] and absent.tolist() == [0.0]


def test_uncertain_fronts_give_a_probability_rather_than_a_verdict():
    # given
    u = np.array([1.0])
    wide = np.array([0.01])

    # when
    prio, _ = lane_priority(u, wide, np.ones(1), u, wide, np.ones(1), blue=True)

    # then
    assert np.isclose(prio[0], 0.5, atol=1e-6)


def test_progress_moments_follow_the_mixture_and_flag_the_unknown():
    # given
    points = np.array([[0.0, 3000.0, 3000.0], [1.0, 9000.0, 9000.0]])

    # when
    mean, var, known = progress_moments(bridge_at(points, np.array([0.0, 0.5, 1.0]), DEFAULT_TABLE))
    _, _, empty = progress_moments(bridge_at(np.zeros((0, 3)), np.array([0.5]), DEFAULT_TABLE))

    # then
    assert np.allclose(mean, [0.4, 0.8, 1.2])
    assert var[0] == 0.0 and var[2] == 0.0 and var[1] > 0.0
    assert known.all() and not empty.any()


def test_band_weights_are_one_hot_when_certain_and_spread_when_not():
    # given
    mean = np.array([1.0, 1.0, 0.5])
    var = np.array([0.0, 0.05**2, 0.0])

    # when
    weights = band_weights(mean, var)

    # then
    assert weights.shape == (3, 5)
    assert weights[0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert 0.5 < weights[1, 2] < 1.0 and weights[1, 1] > 0 and weights[1, 3] > 0
    assert weights[2, 0] == 1.0


def test_lane_counts_spread_a_laners_ticks_over_the_joint_cells_and_sum_to_the_tick_count():
    # given
    ticks = 90
    seat = {
        "u_mean": np.full((10, ticks), 1.1), "u_var": np.zeros((10, ticks)),
        "in_lane": np.zeros((10, ticks)), "alive": np.ones((10, ticks), bool), "farmed": np.zeros((10, ticks)),
    }
    seat["in_lane"][0] = 0.8
    seat["farmed"][0] = 6.0
    lane = {"opp_u": np.full((10, ticks), 0.9), "opp_var": np.zeros((10, ticks)), "opp_presence": np.ones((10, ticks))}
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2
    blue_of = [1] * 5 + [0] * 5

    # when
    rows = lane_counts(seat, lane, blue_of, roles, np.zeros(ticks, bool))
    top = {(situation, outcome): count for s, situation, outcome, count in rows if s == 0}

    # then
    assert np.isclose(sum(c for (situation, _), c in top.items() if situation == "all"), ticks)
    assert np.isclose(top[("all", "theirs_own_farm")], 0.8 * ticks)
    assert np.isclose(top[("all", "off_lane")], 0.2 * ticks)
    assert not any(situation == "pre_objective" for (situation, _) in top)
    assert len(PRIORITY_COLUMNS) == len(SITUATIONS) * len(OUTCOMES) == 124
