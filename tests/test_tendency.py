import numpy as np
import pandas as pd

from synergy.features.tendency import CELLS, cell_of, leave_one_out_ratio, with_priority


def test_cells_name_which_lanes_hold_priority_by_half():
    # given
    top = pd.Series([0.9, 0.1, 0.7, 0.2])
    mid = pd.Series([0.9, 0.1, 0.1, 0.2])
    bot = pd.Series([0.9, 0.1, 0.8, 0.2])
    own_half = pd.Series([1.0, 0.0, 1.0, 0.0])

    # when
    cells = cell_of(top, mid, bot, own_half)

    # then
    assert cells.tolist() == ["tmb_own", "---_away", "t-b_own", "---_away"]
    assert set(cells) <= set(CELLS) and len(CELLS) == 16


def test_the_lagged_priority_joins_onto_the_decision_minute():
    # given
    opportunities = pd.DataFrame({"match_id": ["m", "m"], "puuid": ["p", "p"], "minute": [5, 9], "kind": ["dive", "dive"], "outcome": [1, 0]})
    context = pd.DataFrame({"match_id": ["m"], "puuid": ["p"], "minute": [4], "lane_priority": [0.9], "top_priority": [0.9], "mid_priority": [0.3], "bot_priority": [0.7]})

    # when
    joined = with_priority(opportunities, context)

    # then
    assert np.isclose(joined.loc[0, "lane_priority"], 0.9)
    assert np.isclose(joined.loc[1, "top_priority"], 0.9)
    assert np.isclose(joined.loc[1, "bot_priority"], 0.7)


def test_leave_one_out_ratio_excludes_the_match_and_sits_at_the_prior_without_evidence():
    # given
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m5"],
            "puuid": ["a", "a", "a", "a"],
            "position": ["TOP", "TOP", "TOP", "JUNGLE"],
            "cell": ["tmb_own", "tmb_own", "---_away", "tmb_own"],
            "observed": [2.0, 0.0, 1.0, 9.0],
            "expected": [0.5, 0.5, 0.5, 0.5],
            "variance": [0.2, 0.2, 0.2, 0.2],
        }
    )
    seats = pd.DataFrame(
        {"match_id": ["m1", "m2", "m3", "m4", "m5"], "puuid": ["a", "a", "a", "b", "a"], "position": ["TOP", "TOP", "TOP", "TOP", "JUNGLE"]}
    )

    # when
    out = leave_one_out_ratio(frame, seats, prior=1.0, column="tend_dive").set_index(["match_id", "puuid"])

    # then
    assert np.isclose(out.loc[("m1", "a"), "tend_dive_tmb_own"], np.log((0.0 + 1.0) / (0.5 + 1.0)))
    assert np.isclose(out.loc[("m2", "a"), "tend_dive_tmb_own"], np.log((2.0 + 1.0) / (0.5 + 1.0)))
    assert np.isclose(out.loc[("m1", "a"), "tend_dive_---_away"], np.log((1.0 + 1.0) / (0.5 + 1.0)))
    assert out.loc[("m3", "a"), "tend_dive_---_away"] == 0.0
    assert out.loc[("m4", "b"), "tend_dive_tmb_own"] == 0.0
    assert out.loc[("m5", "a"), "tend_dive_tmb_own"] == 0.0
    assert "position" not in out.columns
