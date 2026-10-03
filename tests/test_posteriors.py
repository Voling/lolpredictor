import numpy as np
import pandas as pd

from synergy.ml import serving
from synergy.ml.posteriors import differences, situations_from, summarise, top_situations
from synergy.ml.serving import adopt_player, exposure_of, write_vectors


def _fit(prefix, situations, outcomes, worlds, kappa, positions=("TOP", "JUNGLE"), states=("all",), gap=True):
    return {
        "prefix": prefix,
        "situations": situations,
        "states": list(states),
        "outcomes": outcomes,
        "worlds": np.array(worlds, dtype=float).reshape(len(situations), len(positions), len(states), len(outcomes)),
        "kappa": np.array(kappa, dtype=float),
        "unit": 1.0,
        "positions": list(positions),
        "typical": np.ones(len(situations)),
        "gap": gap,
    }


def test_a_situation_becomes_a_dirichlet_from_the_served_gap_and_the_chances():
    # given
    columns = ["rsp_kill_ours_far_converged", "rsp_kill_ours_far_held", "tend_follow_-m-_own", "habit_1"]
    standard = {"columns": np.array(columns), "centre": np.array([0.0, 0.0, 0.0, 0.0]), "spread": np.array([0.1, 0.1, 1.0, 1.0])}
    fits = [_fit("rsp", ["kill_ours_far"], ["converged", "held"], [[[0.4, 0.6], [0.5, 0.5]]], [8.0])]
    exposure = {"rsp_kill_ours_far": 12.0, "tend_follow_-m-_own_obs": 9.0, "tend_follow_-m-_own_exp": 4.0}

    # when
    entries = situations_from([2.0, -2.0, 1.5, 0.3], columns, exposure, "TOP", fits, {"follow": 2.0}, standard)
    shares = next(entry for entry in entries if entry["kind"] == "shares")
    ratio = next(entry for entry in entries if entry["kind"] == "ratio")
    summary = summarise(shares, np.random.default_rng(1))
    rate = summarise(ratio, np.random.default_rng(1))

    # then
    assert np.allclose(shares["alpha"], [12.0, 8.0]) and np.allclose(shares["prior"], [0.4, 0.6]) and shares["n"] == 12.0
    assert shares["words"] == "after an ally kill from far away" and ratio["words"] == "following into fights with mid holding priority, on own half"
    assert [outcome["words"] for outcome in summary["outcomes"]] == ["converged", "held ground"]
    first = summary["outcomes"][0]
    assert first["low"] < first["mean"] == 0.6 < first["high"] and first["prior"] == 0.4
    assert rate["ratio"]["mean"] == round(11.0 / 6.0, 3) and rate["ratio"]["low"] < rate["ratio"]["mean"] < rate["ratio"]["high"] and rate["observed"] == 9.0


def test_the_prior_is_a_typical_player_facing_the_same_states_and_the_served_gap_moves_off_it():
    # given
    from scipy import stats

    from synergy.features.reaction import STATES
    from synergy.ml.posteriors import population_percentiles

    columns = ["obj_DRAGON_ours_far_fought", "obj_DRAGON_ours_far_absent"]
    worlds = np.full((1, 1, len(STATES), 2), 0.5)
    worlds[0, 0, STATES.index("early_behind")] = [0.2, 0.8]
    worlds[0, 0, STATES.index("late_ahead")] = [0.6, 0.4]
    fits = {
        "cells": [_fit("obj", ["DRAGON_ours_far"], ["fought", "absent"], worlds, [10.0], ["TOP"], STATES)],
        "standard": {"columns": np.array(columns), "centre": np.zeros(2), "spread": np.full(2, 0.05)},
        "priors": {},
    }
    mixed = {"obj_DRAGON_ours_far@early_behind": 20.0, "obj_DRAGON_ours_far@late_ahead": 20.0}
    ahead = {"obj_DRAGON_ours_far@late_ahead": 40.0}
    z = np.array([2.0, -2.0])

    # when
    first = situations_from(z, columns, mixed, "TOP", fits["cells"], {}, fits["standard"])[0]
    second = situations_from(z, columns, ahead, "TOP", fits["cells"], {}, fits["standard"])[0]
    percentiles = population_percentiles(z, columns, "TOP", fits, None, mixed)
    unseen = population_percentiles(z, columns, "TOP", fits, None, {"obj_DRAGON_ours_far": 40.0})

    # then
    assert np.allclose(first["prior"], [0.4, 0.6]) and np.allclose(first["alpha"], [25.0, 25.0]) and first["n"] == 40.0
    assert np.allclose(second["prior"], [0.6, 0.4]) and np.allclose(second["alpha"], [35.0, 15.0])
    assert np.isclose(percentiles[0], 100.0 * stats.beta.cdf(0.5, 4.0, 6.0)) and np.isnan(unseen).all()


def test_an_old_run_reads_share_cells_against_the_position_world_from_plain_exposure():
    # given
    from scipy import stats

    from synergy.ml.posteriors import population_percentiles

    columns = ["obj_DRAGON_ours_far_fought", "obj_DRAGON_ours_far_absent"]
    fits = {
        "cells": [_fit("obj", ["DRAGON_ours_far"], ["fought", "absent"], [[[0.4, 0.6]]], [10.0], ["TOP"], gap=False)],
        "standard": {"columns": np.array(columns), "centre": np.full(2, 0.5), "spread": np.full(2, 0.1)},
        "priors": {},
    }
    z = np.array([1.0, -1.0])

    # when
    entry = situations_from(z, columns, {"obj_DRAGON_ours_far": 30.0}, "TOP", fits["cells"], {}, fits["standard"])[0]
    percentiles = population_percentiles(z, columns, "TOP", fits)

    # then
    assert np.allclose(entry["prior"], [0.4, 0.6]) and np.allclose(entry["alpha"], [24.0, 16.0]) and entry["n"] == 30.0
    assert np.isclose(percentiles[0], 100.0 * stats.beta.cdf(0.6, 4.0, 6.0))


def test_the_clearest_differences_come_first_and_the_lane_state_collapses_to_where_the_player_stands():
    # given
    from synergy.features.priority import OUTCOMES

    columns = [*(f"prio_all_{outcome}" for outcome in OUTCOMES), "rsp_kill_ours_far_converged", "rsp_kill_ours_far_held"]
    width = len(columns)
    standard = {"columns": np.array(columns), "centre": np.zeros(width), "spread": np.full(width, 0.05)}
    fits = [
        _fit("prio", ["all"], OUTCOMES, np.full((1, 1, len(OUTCOMES)), 1.0 / len(OUTCOMES)), [5.0], ["TOP"]),
        _fit("rsp", ["kill_ours_far"], ["converged", "held"], [[[0.5, 0.5]]], [8.0], ["TOP"]),
    ]
    z = np.zeros(width)
    z[0], z[-2], z[-1] = 2.0, -3.0, 3.0

    # when
    entries = situations_from(z, columns, {"prio_all": 300.0, "rsp_kill_ours_far": 20.0}, "TOP", fits, {}, standard)
    ranked = top_situations(entries, np.random.default_rng(0), top=2)
    lane = next(entry for entry in entries if entry["situation"] == "prio_all")
    typical = top_situations(situations_from(np.zeros(width), columns, {"prio_all": 300.0, "rsp_kill_ours_far": 20.0}, "TOP", fits, {}, standard), np.random.default_rng(0))

    # then
    assert [entry["situation"] for entry in ranked] == ["rsp_kill_ours_far", "prio_all"]
    assert lane["outcomes"] == ["deep_own", "own", "mid", "theirs", "deep_theirs", "off_lane", "dead"]
    assert np.isclose(lane["alpha"].sum(), 305.0) and np.isclose(lane["prior"].sum(), 1.0) and lane["words"] == "lane state"
    assert typical == [] and ranked[0]["own"] == round(20.0 / 28.0, 3)


def test_the_duo_differences_rank_shared_situations_by_how_little_the_posteriors_overlap():
    # given
    alike = {"situation": "rsp_kill_ours_far", "words": "w", "kind": "shares", "n": 30, "alpha": np.array([20.0, 10.0]), "prior": np.array([0.5, 0.5]), "outcomes": ["converged", "held"]}
    apart = {**alike, "situation": "rsp_kill_theirs_far", "alpha": np.array([27.0, 3.0])}
    left = [alike, apart, {"situation": "tend_x", "kind": "ratio"}]
    right = [{**alike, "alpha": np.array([19.0, 11.0])}, {**apart, "alpha": np.array([3.0, 27.0])}]

    # when
    found = differences(left, right, np.random.default_rng(0))

    # then
    assert [entry["situation"] for entry in found] == ["rsp_kill_theirs_far", "rsp_kill_ours_far"]
    assert found[0]["overlap"] < 0.4 < 0.8 < found[1]["overlap"]
    assert found[0]["outcomes"][0]["left"]["mean"] == 0.9 and found[0]["outcomes"][0]["right"]["mean"] == 0.1 and found[0]["outcomes"][1]["words"] == "held ground"


def test_exposure_travels_with_the_pack_and_with_adopted_players(serving_settings):
    # given
    settings = serving_settings
    pd.DataFrame({"puuid": ["a", "b"], "position": ["TOP", "JUNGLE"], "rsp_kill_ours_far": [12.0, 3.0], "tend_dive_tmb_own_obs": [2.0, 0.0]}).to_parquet(settings.processed_dir / serving.EXPOSURE_TABLE, index=False)
    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    adopted = {
        "columns": columns,
        "position": ["TOP"],
        "seats": np.array([4]),
        "evidence": np.array([0.5]),
        "matrix": np.array([[0.0, 1.0]], dtype=np.float32),
        "exposure": np.array([[7.0]], dtype=np.float32),
        "exposure_columns": ["rsp_kill_ours_far"],
    }

    # when
    written = write_vectors(settings.model_dir, settings.processed_dir)
    pack = serving._read_vectors(written)
    adopt_player(settings, {"puuid": "new", "champions": {}}, adopted)

    # then
    assert exposure_of(pack, "a", "TOP") == {"rsp_kill_ours_far": 12.0, "tend_dive_tmb_own_obs": 2.0}
    assert exposure_of(pack, "c", "MIDDLE") == {"rsp_kill_ours_far": 0.0, "tend_dive_tmb_own_obs": 0.0}
    assert exposure_of(serving._vectors(settings, columns), "new", "TOP") == {"rsp_kill_ours_far": 7.0}


def test_the_ranking_prefers_posteriors_that_clearly_leave_the_prior_over_single_events_on_a_rigid_cell():
    # given
    from synergy.ml.posteriors import evidence

    rigid = {"situation": "tend_follow_-m-_own", "kind": "ratio", "observed": 1.0, "expected": 0.0, "prior": 10000.0}
    loose = {"situation": "tend_greed_punished_t-b_away", "kind": "ratio", "observed": 12.0, "expected": 7.4, "prior": 16.8}
    shares = {"situation": "rsp_kill_ours_far", "kind": "shares", "n": 111.0, "alpha": np.array([50.0, 7.0, 54.0]), "prior": np.array([0.39, 0.09, 0.52]), "outcomes": ["died", "fought", "absent"]}

    # when
    rng = np.random.default_rng(3)
    scores = {entry["situation"]: evidence(entry, rng) for entry in (rigid, loose, shares)}

    # then
    assert scores["tend_follow_-m-_own"] < 0.2 < scores["tend_greed_punished_t-b_away"] < scores["rsp_kill_ours_far"]


def _population_fits():
    from synergy.features.priority import OUTCOMES

    columns = ["obj_DRAGON_ours_far_absent", "obj_DRAGON_ours_far_rotated", "obj_DRAGON_ours_far_committed", "prio_all_own_own_farm", "tend_dive_tmb_own"]
    worlds = np.full((1, 1, len(OUTCOMES)), 0.5 / (len(OUTCOMES) - 1))
    worlds[0, 0, OUTCOMES.index("own_own_farm")] = 0.5
    return columns, {
        "cells": [
            _fit("obj", ["DRAGON_ours_far"], ["absent", "rotated", "committed"], [[[0.505, 0.49, 0.005]]], [78.4], ["TOP"]),
            _fit("prio", ["all"], OUTCOMES, worlds, [3.0], ["TOP"]),
        ],
        "standard": {"columns": np.array(columns), "centre": np.zeros(len(columns)), "spread": np.ones(len(columns))},
        "priors": {"dive": 59.5},
    }


def test_a_reading_is_ranked_against_the_fitted_spread_of_true_rates_not_against_shrunk_estimates():
    # given
    from scipy import stats

    from synergy.ml.posteriors import population_percentiles

    columns, fits = _population_fits()
    z = np.array([0.035, -0.19, -0.001, 0.5, np.log(1.2)])

    # when
    found = population_percentiles(z, columns, "TOP", fits, None, {"obj_DRAGON_ours_far": 25.0})

    # then
    assert 70.0 < found[0] < 76.0 and np.isnan(found[2]) and np.isnan(found[3])
    assert np.isclose(found[4], 100.0 * stats.gamma.cdf(1.2, 59.5, scale=1.0 / 59.5))


def test_lane_state_is_ranked_only_among_well_measured_players_and_only_for_a_well_measured_player():
    # given
    from synergy.ml.posteriors import population_percentiles

    columns, fits = _population_fits()
    games = np.array([30.0] * 100 + [2.0] * 100)
    values = np.concatenate([np.linspace(0.40, 0.60, 100), np.full(100, 0.9)])
    matrix = np.zeros((200, len(columns)), dtype=np.float32)
    matrix[:, 3] = values
    pack = {"position": np.array(["TOP"] * 200), "matrix": matrix, "exposure": games[:, None].astype(np.float16), "exposure_columns": ["prio_all"]}
    z = np.array([0.5, 0.49, 0.005, 0.55, 0.0])

    # when
    measured = population_percentiles(z, columns, "TOP", fits, pack, {"prio_all": 48.0})
    thin = population_percentiles(z, columns, "TOP", fits, pack, {"prio_all": 6.0})

    # then
    assert measured[3] == 75.0 and np.isnan(thin[3])


def test_each_chart_says_what_the_player_usually_does_and_what_is_unusual():
    # given
    from synergy.ml.posteriors import duo_takeaway, ratio_takeaway, share_takeaway

    outcomes = ["died", "fought", "rotated", "absent"]
    prior = np.array([0.19, 0.47, 0.01, 0.33])
    mean = np.array([0.04, 0.60, 0.01, 0.35])
    draws = np.random.default_rng(0).dirichlet(mean * 300, 400)

    # when
    usual = share_takeaway("obj_HORDE_ours_near", outcomes, mean, prior, draws, "Weaver", "jungler")
    typical = share_takeaway("obj_HORDE_ours_near", outcomes, prior, prior, np.random.default_rng(0).dirichlet(prior * 300, 400), "Weaver", "jungler")
    ratio = ratio_takeaway("tend_dive_---_away", 2.1, 1.4, 3.0, "Gakgos", "top laner")
    duo = duo_takeaway("obj_DRAGON_ours_far", outcomes, np.array([0.2, 0.43, 0.33, 0.04]), np.array([0.39, 0.08, 0.01, 0.52]), ("Gryffinn", "Gakgos"))
    shared = duo_takeaway("obj_DRAGON_ours_far", outcomes, np.array([0.1, 0.1, 0.2, 0.6]), np.array([0.3, 0.05, 0.05, 0.6]), ("A", "B"))

    # then
    assert usual == "At ally void grubs nearby, Weaver usually fights, 60%, and dies far less often than a typical jungler in the same spots: 4% against 19%."
    assert typical == "At ally void grubs nearby, Weaver usually fights, 47%, like a typical jungler in the same spots."
    assert ratio == "Gakgos dives with no lane holding priority, on the enemy half, 2.1 times as often as a typical top laner in the same spots."
    assert duo == "At an ally dragon from far away, expect Gryffinn to fight, 43%, and Gakgos to not join, 52%."
    assert shared == "At an ally dragon from far away, you both usually don't join, but B dies more often: 30% against 10%."


def test_situations_whose_events_arrive_in_clusters_are_left_out_of_charts_and_standouts():
    # given
    from synergy.ml.posteriors import population_percentiles

    columns = ["obj_HORDE_ours_near_fought", "obj_HORDE_ours_near_absent", "obj_DRAGON_ours_near_fought", "obj_DRAGON_ours_near_absent"]
    fits = {
        "cells": [_fit("obj", ["HORDE_ours_near", "DRAGON_ours_near"], ["fought", "absent"], [[[0.4, 0.6]], [[0.4, 0.6]]], [1.9, 115.4], ["JUNGLE"])],
        "standard": {"columns": np.array(columns), "centre": np.zeros(4), "spread": np.ones(4)},
        "priors": {},
    }
    z = np.array([0.3, -0.3, 0.3, -0.3])
    exposure = {"obj_HORDE_ours_near": 30.0, "obj_DRAGON_ours_near": 30.0}

    # when
    entries = situations_from(z, columns, exposure, "JUNGLE", fits["cells"], {}, fits["standard"])
    percentiles = population_percentiles(z, columns, "JUNGLE", fits, None, exposure)

    # then
    assert [entry["situation"] for entry in entries] == ["obj_DRAGON_ours_near"]
    assert np.isnan(percentiles[:2]).all() and not np.isnan(percentiles[2:]).any()


def test_a_gold_reading_draws_against_a_typical_player_in_the_same_gold_state_and_needs_its_columns():
    # given
    from scipy import stats

    from synergy.features.reaction import GOLD_SPLIT, STATES
    from synergy.ml.posteriors import population_percentiles

    worlds = np.full((1, 1, len(STATES), 2), 0.5)
    worlds[0, 0, STATES.index("early_behind")] = [0.2, 0.8]
    worlds[0, 0, STATES.index("late_behind")] = [0.3, 0.7]
    fit = {**_fit("rsp", ["plate_ours_far"], ["converged", "absent"], worlds, [10.0], ["TOP"], STATES), "split": {**GOLD_SPLIT, "situations": ["plate_ours_far"]}}
    columns = ["rsp_plate_ours_far_converged", "rsp_plate_ours_far_absent", "rspg_behind_plate_ours_far_converged", "rspg_behind_plate_ours_far_absent"]
    fits = {"cells": [fit], "standard": {"columns": np.array(columns), "centre": np.zeros(4), "spread": np.full(4, 0.05)}, "priors": {}}
    exposure = {"rsp_plate_ours_far@early_behind": 40.0, "rsp_plate_ours_far@late_behind": 40.0, "rsp_plate_ours_far@late_ahead": 40.0}
    z = np.array([0.0, 0.0, 2.0, -2.0])

    # when
    entries = {entry["situation"]: entry for entry in situations_from(z, columns, exposure, "TOP", [fit], {}, fits["standard"])}
    behind = summarise(entries["rspg_behind_plate_ours_far"], np.random.default_rng(0), "bblskibs", "top laner")
    percentiles = population_percentiles(z, columns, "TOP", fits, None, exposure)
    without = situations_from(z[:2], columns[:2], exposure, "TOP", [fit], {}, fits["standard"])

    # then
    assert sorted(entries) == ["rsp_plate_ours_far", "rspg_behind_plate_ours_far"] and [entry["situation"] for entry in without] == ["rsp_plate_ours_far"]
    assert np.allclose(entries["rsp_plate_ours_far"]["prior"], [1.0 / 3.0, 2.0 / 3.0]) and entries["rsp_plate_ours_far"]["n"] == 120.0
    assert np.allclose(entries["rspg_behind_plate_ours_far"]["prior"], [0.25, 0.75]) and np.allclose(entries["rspg_behind_plate_ours_far"]["alpha"], [31.5, 58.5])
    assert entries["rspg_behind_plate_ours_far"]["n"] == 80.0 and entries["rspg_behind_plate_ours_far"]["words"] == "when behind, after an ally plate from far away"
    assert np.isclose(percentiles[2], 100.0 * stats.beta.cdf(0.35, 2.5, 7.5))
    assert behind["takeaway"] == "When behind, after an ally plate from far away, bblskibs usually does not follow, 65%, and moves in more often than a typical top laner who is behind: 35% against 25%."
