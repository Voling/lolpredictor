import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from synergy.features.cells import SeatIndex, buffer, cells_frame, dense_counts, evidence_shares, fit_cells, moment_kappa, write_cells


def _cells(counts, seats, situations, outcomes, prefix, scale=1.0):
    index = SeatIndex(seats)
    dense = dense_counts(counts, index, situations, outcomes)
    fit = fit_cells(dense, index, situations, outcomes, prefix, scale)
    return cells_frame([(fit, dense)], index, fit.columns), fit.report


def test_cells_are_gaps_from_the_position_world_that_leave_the_match_out_and_are_zero_without_other_games():
    # given
    counts = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m2", "m3"],
            "puuid": ["a", "a", "a", "a", "b"],
            "situation": ["s", "s", "s", "s", "s"],
            "outcome": ["x", "y", "x", "x", "y"],
            "count": [3.0, 1.0, 2.0, 2.0, 5.0],
        }
    )
    seats = pd.DataFrame({"match_id": ["m1", "m2", "m3", "m4"], "puuid": ["a", "a", "b", "c"], "position": ["TOP"] * 4})

    # when
    block, report = _cells(counts, seats, ["s"], ["x", "y"], "t")
    block = block.set_index(["match_id", "puuid"])

    # then
    assert np.allclose(block[["t_s_x", "t_s_y"]].sum(axis=1), 0.0)
    assert block.loc[("m1", "a"), "t_s_x"] > block.loc[("m3", "b"), "t_s_x"] == 0.0
    assert np.allclose(block.loc[("m4", "c"), ["t_s_x", "t_s_y"]].to_numpy(), 0.0)
    world = np.array(report["s"]["world"]["TOP"]["all"])
    kappa = report["s"]["kappa"]
    assert np.isclose(block.loc[("m1", "a"), "t_s_x"], (4.0 - 4.0 * world[0]) / (4.0 + kappa), atol=1e-3)


def test_a_player_is_pooled_per_position_and_measured_against_that_position_world():
    # given
    rows = []
    for match in range(6):
        rows.append({"match_id": f"t{match}", "puuid": "a", "situation": "s", "outcome": "x", "count": 9.0})
        rows.append({"match_id": f"t{match}", "puuid": "a", "situation": "s", "outcome": "y", "count": 1.0})
        rows.append({"match_id": f"c{match}", "puuid": "c", "situation": "s", "outcome": "x", "count": 1.0})
        rows.append({"match_id": f"c{match}", "puuid": "c", "situation": "s", "outcome": "y", "count": 9.0})
        rows.append({"match_id": f"j{match}", "puuid": "b", "situation": "s", "outcome": "y", "count": 10.0})
    rows.append({"match_id": "j_a1", "puuid": "a", "situation": "s", "outcome": "y", "count": 2.0})
    rows.append({"match_id": "j_a2", "puuid": "a", "situation": "s", "outcome": "y", "count": 2.0})
    counts = pd.DataFrame(rows)
    seats = pd.DataFrame(
        {
            "match_id": [f"t{i}" for i in range(6)] + [f"c{i}" for i in range(6)] + [f"j{i}" for i in range(6)] + ["j_a1", "j_a2"],
            "puuid": ["a"] * 6 + ["c"] * 6 + ["b"] * 6 + ["a", "a"],
            "position": ["TOP"] * 12 + ["JUNGLE"] * 8,
        }
    )

    # when
    block, report = _cells(counts, seats, ["s"], ["x", "y"], "t")
    block = block.set_index(["match_id", "puuid"])

    # then
    assert 0.4 < report["s"]["world"]["TOP"]["all"][0] < 0.6 and report["s"]["world"]["JUNGLE"]["all"][0] < 0.05
    assert block.loc[("t0", "a"), "t_s_x"] > 0.2 and block.loc[("c0", "c"), "t_s_x"] < -0.2
    assert abs(block.loc[("j_a1", "a"), "t_s_x"]) < 0.01


def test_a_situation_nobody_met_yields_no_gap_for_anyone():
    # given
    counts = pd.DataFrame({"match_id": ["m1"], "puuid": ["a"], "situation": ["seen"], "outcome": ["x"], "count": [1.0]})
    seats = pd.DataFrame({"match_id": ["m1", "m2"], "puuid": ["a", "b"], "position": ["MIDDLE", "MIDDLE"]})

    # when
    block, report = _cells(counts, seats, ["seen", "unseen"], ["x", "y"], "t")

    # then
    assert set(block.columns) == {"match_id", "puuid", "t_seen_x", "t_seen_y", "t_unseen_x", "t_unseen_y"}
    assert np.allclose(block[["t_unseen_x", "t_unseen_y"]].to_numpy(), 0.0)
    assert report["unseen"]["rows"] == 0 and report["unseen"]["world"]["MIDDLE"]["all"] == [0.5, 0.5]


def test_counts_land_in_the_state_of_their_chance_and_unknown_states_are_dropped():
    # given
    seats = pd.DataFrame({"match_id": ["m1", "m2"], "puuid": ["a", "a"], "position": ["TOP", "TOP"]})
    counts = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m2"],
            "puuid": ["a"] * 4,
            "situation": ["s"] * 4,
            "state": ["early", "late", "late", "never"],
            "outcome": ["x", "y", "y", "x"],
            "count": [1.0] * 4,
        }
    )
    index = SeatIndex(seats)

    # when
    dense = dense_counts(counts, index, ["s"], ["x", "y"], states=["early", "late"])
    single = dense_counts(counts, index, ["s"], ["x", "y"])

    # then
    first, second = index.rows(["m1", "m2"], ["a", "a"])
    assert dense.shape == (2, 1, 2, 2) and single.shape == (2, 1, 1, 2)
    assert dense[first, 0].tolist() == [[1.0, 0.0], [0.0, 1.0]] and dense[second, 0].tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert single[second, 0, 0].tolist() == [1.0, 1.0]


def test_a_cell_is_the_shrunk_gap_between_the_player_and_a_typical_player_facing_the_same_states():
    # given
    rows = [
        ("g1", "a", "early", 3.0, 0.0),
        ("g2", "a", "late", 0.0, 4.0),
        ("g3", "a", "early", 1.0, 1.0),
        ("h1", "b", "early", 0.0, 2.0),
        ("h2", "b", "late", 3.0, 1.0),
    ]
    counts = pd.DataFrame(
        [
            {"match_id": match, "puuid": puuid, "situation": "s", "state": state, "outcome": outcome, "count": count}
            for match, puuid, state, x, y in rows
            for outcome, count in (("x", x), ("y", y))
        ]
    )
    seats = pd.DataFrame({"match_id": ["g1", "g2", "g3", "h1", "h2", "k1"], "puuid": ["a", "a", "a", "b", "b", "c"], "position": ["TOP"] * 6})
    index = SeatIndex(seats)
    dense = dense_counts(counts, index, ["s"], ["x", "y"], states=["early", "late"])

    # when
    fit = fit_cells(dense, index, ["s"], ["x", "y"], "t", kappa=[4.0], states=["early", "late"])
    block = cells_frame([(fit, dense)], index, fit.columns).set_index(["match_id", "puuid"])

    # then
    early, late = np.array([5.0, 4.0]) / 9.0, np.array([4.0, 6.0]) / 10.0
    assert np.allclose(fit.worlds[0, 0], [early, late])
    mix = (2.0 * early + 4.0 * late) / 6.0
    shrunk = (4.0 * mix + np.array([1.0, 5.0])) / (6.0 + 4.0)
    assert np.allclose(block.loc[("g1", "a"), ["t_s_x", "t_s_y"]].to_numpy(dtype=float), shrunk - mix)
    assert np.allclose(block.loc[("h1", "b"), ["t_s_x", "t_s_y"]].to_numpy(dtype=float), (np.array([3.0, 1.0]) - 4.0 * late) / (4.0 + 4.0))
    assert np.allclose(block.loc[("k1", "c"), ["t_s_x", "t_s_y"]].to_numpy(dtype=float), 0.0)


def test_counts_streamed_into_a_disk_buffer_in_pieces_match_one_pass_and_write_the_same_cells(tmp_path):
    # given
    rng = np.random.default_rng(3)
    seats = pd.DataFrame(
        {"match_id": [f"m{i // 4}" for i in range(80)], "puuid": [f"p{i % 12}" for i in range(80)], "position": [["TOP", "MIDDLE"][i % 2] for i in range(80)]}
    )
    counts = pd.DataFrame(
        {
            "match_id": seats["match_id"].sample(400, replace=True, random_state=1).to_numpy(),
            "puuid": seats["puuid"].sample(400, replace=True, random_state=2).to_numpy(),
            "situation": rng.choice(["s", "t", "elsewhere"], 400),
            "outcome": rng.choice(["x", "y", "z"], 400),
            "count": rng.integers(1, 5, 400).astype(float),
        }
    )
    index = SeatIndex(seats)

    # when
    whole = dense_counts(counts, index, ["s", "t"], ["x", "y"])
    streamed = buffer(tmp_path / "buffers" / "counts.npy", index, ["s", "t"], ["x", "y"])
    for piece in np.array_split(np.arange(len(counts)), 3):
        dense_counts(counts.iloc[piece], index, ["s", "t"], ["x", "y"], out=streamed)
    fit = fit_cells(streamed, index, ["s", "t"], ["x", "y"], "c")
    write_cells(tmp_path / "cells.parquet", [(fit, streamed)], index, fit.columns)
    written = pq.read_table(tmp_path / "cells.parquet").to_pandas()
    shares = evidence_shares(fit, index)

    # then
    assert np.allclose(np.asarray(streamed), whole)
    assert written.equals(cells_frame([(fit, whole)], index, fit.columns))
    assert len(shares) == len(index.who) and shares["share"].between(0.0, 1.0).all()


def _seats_and_counts(distinct: bool):
    rng = np.random.default_rng(9)
    rows, seats = [], []
    for player in range(60):
        lean = 0.9 if (distinct and player % 2 == 0) else (0.1 if distinct else 0.5)
        for game in range(8):
            match = f"m{player}_{game}"
            seats.append({"match_id": match, "puuid": f"p{player}", "position": "TOP"})
            share = np.clip(rng.normal(lean, 0.05), 0.0, 1.0)
            rows.append({"match_id": match, "puuid": f"p{player}", "situation": "s", "outcome": "x", "count": 90.0 * share})
            rows.append({"match_id": match, "puuid": f"p{player}", "situation": "s", "outcome": "y", "count": 90.0 * (1.0 - share)})
    return pd.DataFrame(rows), pd.DataFrame(seats)


def test_the_moment_concentration_floors_at_three_games_when_players_differ_and_grows_when_they_do_not():
    # given
    apart, apart_seats = _seats_and_counts(distinct=True)
    alike, alike_seats = _seats_and_counts(distinct=False)

    # when
    apart_index = SeatIndex(apart_seats)
    alike_index = SeatIndex(alike_seats)
    small = moment_kappa(dense_counts(apart, apart_index, ["s"], ["x", "y"]), apart_index, 0)
    large = moment_kappa(dense_counts(alike, alike_index, ["s"], ["x", "y"]), alike_index, 0)

    # then
    assert small == 3.0
    assert large > 8.0


def test_a_fixed_concentration_per_situation_is_used_as_given():
    # given
    counts, seats = _seats_and_counts(distinct=True)
    index = SeatIndex(seats)
    dense = dense_counts(counts, index, ["s"], ["x", "y"])

    # when
    fit = fit_cells(dense, index, ["s"], ["x", "y"], "t", unit=90.0, kappa=[2.5])

    # then
    assert fit.report["s"]["kappa"] == 2.5 and fit.unit == 90.0


def test_a_split_reading_measures_the_gap_over_only_the_chances_in_its_group_of_states():
    # given
    rows = [
        ("g1", "a", "s", "early_behind", 2.0, 0.0),
        ("g1", "a", "s", "early_ahead", 0.0, 1.0),
        ("g2", "a", "s", "late_behind", 1.0, 3.0),
        ("g2", "a", "t", "early_behind", 1.0, 0.0),
        ("g3", "a", "s", "early_ahead", 2.0, 2.0),
        ("h1", "b", "s", "early_behind", 0.0, 2.0),
        ("h1", "b", "s", "late_behind", 1.0, 1.0),
    ]
    counts = pd.DataFrame(
        [
            {"match_id": match, "puuid": puuid, "situation": situation, "state": state, "outcome": outcome, "count": count}
            for match, puuid, situation, state, go, stay in rows
            for outcome, count in (("go", go), ("stay", stay))
        ]
    )
    seats = pd.DataFrame({"match_id": ["g1", "g2", "g3", "h1"], "puuid": ["a", "a", "a", "b"], "position": ["TOP"] * 4})
    states = ["early_behind", "late_behind", "early_ahead"]
    split = {"prefix": "rg", "situations": ["s"], "groups": {"behind": ["early_behind", "late_behind"], "ahead": ["early_ahead"]}}
    index = SeatIndex(seats)
    dense = dense_counts(counts, index, ["s", "t"], ["go", "stay"], states=states)

    # when
    fit = fit_cells(dense, index, ["s", "t"], ["go", "stay"], "r", kappa=[4.0, 4.0], states=states, split=split)
    block = cells_frame([(fit, dense)], index, fit.columns).set_index(["match_id", "puuid"])

    # then
    late_behind, early_ahead = np.array([3.0, 5.0]) / 8.0, np.array([3.0, 4.0]) / 7.0
    assert fit.columns[-4:] == ["rg_behind_s_go", "rg_behind_s_stay", "rg_ahead_s_go", "rg_ahead_s_stay"]
    assert np.allclose(block.loc[("g1", "a"), ["rg_behind_s_go", "rg_behind_s_stay"]].to_numpy(dtype=float), (np.array([1.0, 3.0]) - 4.0 * late_behind) / (4.0 + 4.0))
    assert np.allclose(block.loc[("g1", "a"), ["rg_ahead_s_go", "rg_ahead_s_stay"]].to_numpy(dtype=float), (np.array([2.0, 2.0]) - 4.0 * early_ahead) / (4.0 + 4.0))
    assert np.allclose(block.loc[("h1", "b"), fit.columns[-4:]].to_numpy(dtype=float), 0.0)
