import numpy as np
import pandas as pd

from synergy.features.cells import SINGLE, SeatIndex
from synergy.features.exposure import exposure_frame, seat_exposure


def test_exposure_sums_each_players_chances_per_situation_across_their_games():
    # given
    seats = pd.DataFrame({"match_id": ["m1", "m1", "m2", "m2"], "puuid": ["a", "b", "a", "b"], "position": ["TOP", "JUNGLE", "TOP", "JUNGLE"]})
    index = SeatIndex(seats)
    counts = np.zeros((4, 2, 3))
    counts[index.rows(["m1"], ["a"])[0]] = [[1, 2, 0], [0, 0, 4]]
    counts[index.rows(["m2"], ["a"])[0]] = [[3, 0, 0], [1, 1, 0]]
    counts[index.rows(["m1"], ["b"])[0]] = [[0, 0, 5], [0, 0, 0]]

    # when
    totals = seat_exposure(counts, index, ["near", "far"], unit=2.0)
    frame = exposure_frame(index, [("rsp", ["near", "far"], SINGLE, totals)])

    # then
    by = frame.set_index(["puuid", "position"])
    assert by.loc[("a", "TOP"), "rsp_near"] == 3.0 and by.loc[("a", "TOP"), "rsp_far"] == 3.0
    assert by.loc[("b", "JUNGLE"), "rsp_near"] == 2.5 and by.loc[("b", "JUNGLE"), "rsp_far"] == 0.0
    assert list(frame.columns) == ["puuid", "position", "rsp_near", "rsp_far"]


def test_exposure_keeps_each_state_apart_so_a_typical_mix_can_be_rebuilt():
    # given
    seats = pd.DataFrame({"match_id": ["m1", "m2", "m1"], "puuid": ["a", "a", "b"], "position": ["TOP", "TOP", "JUNGLE"]})
    index = SeatIndex(seats)
    states = ["early_even", "late_ahead"]
    counts = np.zeros((3, 1, 2, 2))
    counts[index.rows(["m1"], ["a"])[0]] = [[[1, 1], [0, 3]]]
    counts[index.rows(["m2"], ["a"])[0]] = [[[2, 0], [1, 0]]]
    wards = np.ones((3, 1, 4))

    # when
    frame = exposure_frame(
        index,
        [("obj", ["DRAGON_ours_far"], states, seat_exposure(counts, index, ["DRAGON_ours_far"], states=states)), ("ward", ["early"], SINGLE, seat_exposure(wards, index, ["early"]))],
    )

    # then
    by = frame.set_index(["puuid", "position"])
    assert list(frame.columns) == ["puuid", "position", "obj_DRAGON_ours_far@early_even", "obj_DRAGON_ours_far@late_ahead", "ward_early"]
    assert by.loc[("a", "TOP"), "obj_DRAGON_ours_far@early_even"] == 4.0 and by.loc[("a", "TOP"), "obj_DRAGON_ours_far@late_ahead"] == 4.0
    assert by.loc[("b", "JUNGLE"), "obj_DRAGON_ours_far@early_even"] == 0.0 and by.loc[("a", "TOP"), "ward_early"] == 8.0
