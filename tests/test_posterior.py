import numpy as np

from synergy.features.posterior import (
    DEFAULT_TABLE,
    bridge_at,
    dead_mask,
    death_spans,
    fit_sigma,
    known_points,
    nearer_residuals,
    region_mass,
    sigma_at,
    top_regions,
    wave_mass,
)
from synergy.features.regions import REGION_INDEX


def test_the_posterior_is_a_mixture_over_both_frames_weighted_by_time():
    # given
    points = np.array([[0.0, 1000.0, 1000.0], [1.0, 3000.0, 1000.0]])

    # when
    mix = bridge_at(points, np.array([0.0, 0.25, 0.75, 1.0]), DEFAULT_TABLE)

    # then
    assert np.allclose(mix["w"], [1.0, 0.25, 0.75, 1.0])
    assert mix["s1"][0] == 0.0 and mix["s1"][3] == 0.0
    assert mix["s0"][1] < mix["s1"][1]
    assert mix["s0"][2] > mix["s1"][2]
    assert np.allclose(mix["p0"][1], [1000.0, 1000.0]) and np.allclose(mix["p1"][1], [3000.0, 1000.0])


def test_the_filtered_posterior_never_looks_at_the_later_frame():
    # given
    points = np.array([[0.0, 1000.0, 1000.0], [1.0, 3000.0, 1000.0]])

    # when
    smoothed = bridge_at(points, np.array([0.5]), DEFAULT_TABLE)
    filtered = bridge_at(points, np.array([0.5]), DEFAULT_TABLE, filtered=True)

    # then
    assert smoothed["w"][0] == 0.5
    assert filtered["w"][0] == 0.0 and np.allclose(filtered["p0"][0], [1000.0, 1000.0])
    assert filtered["s0"][0] > smoothed["s0"][0] * 0 and np.isinf(filtered["s1"][0])


def test_spread_grows_with_time_from_the_nearer_frame_and_extrapolates():
    # given
    table = DEFAULT_TABLE

    # when
    near, mid, far = sigma_at(table, np.array([0.05, 0.25, 0.8]))

    # then
    assert near < mid < far
    assert far > table[-1]


def test_an_anchor_inside_the_minute_becomes_an_exact_point():
    # given
    frames = np.array([[1000.0, 1000.0], [1000.0, 1000.0]])
    certain = [(0.5, 5000.0, 5000.0)]

    # when
    points = known_points(frames, np.array([True, True]), [], certain, [], team=100)
    mix = bridge_at(points, np.array([0.5]), DEFAULT_TABLE)

    # then
    assert len(points) == 3
    assert mix["w"][0] == 1.0 and mix["s1"][0] == 0.0 and np.allclose(mix["p1"][0], [5000.0, 5000.0])


def test_a_dead_player_has_no_frame_point_and_respawns_at_the_fountain():
    # given
    frames = np.array([[7000.0, 7000.0], [7000.0, 7000.0], [7000.0, 7000.0]])
    spans = death_spans([(0.9, 6)])

    # when
    dead = dead_mask(spans, np.array([0.95, 1.5]))
    points = known_points(frames, np.array([True, True, True]), spans, [], [], team=100)

    # then
    assert dead.tolist() == [True, False]
    assert not any(p[0] == 1.0 for p in points)
    assert any(np.isclose(p[1], 394.0) and np.isclose(p[2], 461.0) for p in points)


def test_region_mass_is_a_distribution_that_splits_across_both_frames():
    # given
    points = np.array([[0.0, 1000.0, 1000.0], [1.0, 13000.0, 2500.0]])
    mix = bridge_at(points, np.array([0.5, 0.0]), DEFAULT_TABLE)

    # when
    masses = region_mass(mix, team=100)

    # then
    assert np.allclose(masses.sum(axis=1), 1.0, atol=1e-4)
    assert masses[0, REGION_INDEX["BASE_OWN"]] > 0.1
    assert masses[0, REGION_INDEX["LANE_BOT_NEUTRAL"]] > 0.1
    assert (masses[0] > 0.01).sum() >= 3
    assert masses[1, REGION_INDEX["BASE_OWN"]] == 1.0


def test_wave_mass_pushes_when_farming_deep_and_is_off_lane_when_dead():
    # given
    points = np.array([[0.0, 13000.0, 2500.0], [1.0, 13000.0, 2500.0]])
    mix = bridge_at(points, np.array([0.0, 0.0]), DEFAULT_TABLE)
    masses = region_mass(mix, team=100)

    # when
    waves = wave_mass(masses, mix, 100, "BOTTOM", np.array([6.0, 6.0]), np.array([False, True]))

    # then
    assert waves[0, 0] == 1.0 and waves[0, 2] == 0.0
    assert waves[1].tolist() == [0.0, 0.0, 1.0]


def test_calibration_measures_the_nearer_frame_and_fills_thin_bins_from_the_default():
    # given
    points = np.array([[0.0, 1000.0, 1000.0], [1.0, 5000.0, 1000.0]])
    anchors = [(0.05, 1300.0, 1000.0)] * 150

    # when
    residuals = {"TOP": nearer_residuals(points, anchors)}
    table = fit_sigma(residuals)

    # then
    assert residuals["TOP"][0] == (0.05, 300.0**2)
    assert np.isclose(table["TOP"][0], 300.0 / np.sqrt(2.0))
    assert table["TOP"][1:] == DEFAULT_TABLE[1:]
    assert table["JUNGLE"] == DEFAULT_TABLE


def test_top_regions_keep_the_largest_masses_in_order():
    # given
    masses = np.array([[0.1, 0.6, 0.3, 0.0]], np.float32)

    # when
    order, picked = top_regions(masses, top=2)

    # then
    assert order.tolist() == [[1, 2]]
    assert np.allclose(picked.astype(float), [[0.6, 0.3]], atol=1e-3)
