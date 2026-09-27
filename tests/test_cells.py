import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from synergy.features.cells import SeatIndex, buffer, cells_frame, dense_counts, evidence_shares, fit_cells, moment_kappa, write_cells


def _cells(counts, seats, situations, outcomes, prefix, scale=1.0):
    index = SeatIndex(seats)
    dense = dense_counts(counts, index, situations, outcomes)
    fit = fit_cells(dense, index, situations, outcomes, prefix, scale)
    return cells_frame([(fit, dense)], index, fit.columns), fit.report


def test_cells_are_distributions_that_leave_the_match_out_and_fall_back_to_the_position_world():
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
    assert np.allclose(block[["t_s_x", "t_s_y"]].sum(axis=1), 1.0)
    assert block.loc[("m1", "a"), "t_s_x"] > block.loc[("m3", "b"), "t_s_x"]
    world = np.array(report["s"]["world"]["TOP"])
    assert np.allclose(block.loc[("m4", "c"), ["t_s_x", "t_s_y"]].to_numpy(), world, atol=1e-3)
    own = block.loc[("m1", "a"), "t_s_x"]
    kappa = report["s"]["kappa"]
    assert np.isclose(own, (4.0 + kappa * world[0]) / (4.0 + kappa), atol=1e-3)


def test_a_player_is_pooled_per_position_and_shrinks_toward_that_position_world():
    # given
    rows = []
    for match in range(6):
        rows.append({"match_id": f"t{match}", "puuid": "a", "situation": "s", "outcome": "x", "count": 9.0})
        rows.append({"match_id": f"t{match}", "puuid": "a", "situation": "s", "outcome": "y", "count": 1.0})
        rows.append({"match_id": f"j{match}", "puuid": "b", "situation": "s", "outcome": "y", "count": 10.0})
    rows.append({"match_id": "j_a", "puuid": "a", "situation": "s", "outcome": "y", "count": 2.0})
    counts = pd.DataFrame(rows)
    seats = pd.DataFrame(
        {
            "match_id": [f"t{i}" for i in range(6)] + [f"j{i}" for i in range(6)] + ["j_a"],
            "puuid": ["a"] * 6 + ["b"] * 6 + ["a"],
            "position": ["TOP"] * 6 + ["JUNGLE"] * 7,
        }
    )

    # when
    block, report = _cells(counts, seats, ["s"], ["x", "y"], "t")
    block = block.set_index(["match_id", "puuid"])

    # then
    assert report["s"]["world"]["TOP"][0] > 0.8 and report["s"]["world"]["JUNGLE"][0] < 0.2
    assert block.loc[("t0", "a"), "t_s_x"] > 0.8
    assert block.loc[("j_a", "a"), "t_s_x"] < 0.2


def test_a_situation_nobody_met_still_yields_the_world_row_for_everyone():
    # given
    counts = pd.DataFrame({"match_id": ["m1"], "puuid": ["a"], "situation": ["seen"], "outcome": ["x"], "count": [1.0]})
    seats = pd.DataFrame({"match_id": ["m1", "m2"], "puuid": ["a", "b"], "position": ["MIDDLE", "MIDDLE"]})

    # when
    block, report = _cells(counts, seats, ["seen", "unseen"], ["x", "y"], "t")

    # then
    assert set(block.columns) == {"match_id", "puuid", "t_seen_x", "t_seen_y", "t_unseen_x", "t_unseen_y"}
    assert np.allclose(block[["t_unseen_x", "t_unseen_y"]].to_numpy(), 0.5)
    assert report["unseen"]["rows"] == 0


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
            "count": rng.random(400),
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
