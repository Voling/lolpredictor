import numpy as np
import pandas as pd

from synergy.features.cells import SeatIndex
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
    frame = exposure_frame(index, [("rsp", ["near", "far"], totals)])

    # then
    by = frame.set_index(["puuid", "position"])
    assert by.loc[("a", "TOP"), "rsp_near"] == 3.0 and by.loc[("a", "TOP"), "rsp_far"] == 3.0
    assert by.loc[("b", "JUNGLE"), "rsp_near"] == 2.5 and by.loc[("b", "JUNGLE"), "rsp_far"] == 0.0
    assert list(frame.columns) == ["puuid", "position", "rsp_near", "rsp_far"]
