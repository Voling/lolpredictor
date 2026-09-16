import numpy as np
import pandas as pd

from synergy.features.cells import cell_block


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
    block, report = cell_block(counts, seats, ["s"], ["x", "y"], "t")
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
    block, report = cell_block(counts, seats, ["s"], ["x", "y"], "t")
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
    block, report = cell_block(counts, seats, ["seen", "unseen"], ["x", "y"], "t")

    # then
    assert set(block.columns) == {"match_id", "puuid", "t_seen_x", "t_seen_y", "t_unseen_x", "t_unseen_y"}
    assert np.allclose(block[["t_unseen_x", "t_unseen_y"]].to_numpy(), 0.5)
    assert report["unseen"]["rows"] == 0
