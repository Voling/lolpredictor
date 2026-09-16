import numpy as np
import pandas as pd

from synergy.features.priority import CONTEXT_COLUMNS, priority_context_from


def test_context_carries_each_lane_of_the_team_and_junglers_take_the_best_lane():
    # given
    track = pd.DataFrame(
        {
            "match_id": ["m"] * 7,
            "puuid": ["top", "top", "mid", "adc", "sup", "jng", "top"],
            "tick": [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 1.0],
            "prio_filtered": [1.0, 0.0, 0.2, 0.6, 0.8, None, 0.9],
        }
    )
    seats = pd.DataFrame(
        {
            "match_id": ["m"] * 5,
            "puuid": ["top", "mid", "adc", "sup", "jng"],
            "team_id": [100] * 5,
            "position": ["TOP", "MIDDLE", "BOTTOM", "UTILITY", "JUNGLE"],
        }
    )

    # when
    out = priority_context_from(track, seats).set_index(["puuid", "minute"])

    # then
    assert list(out.columns) == ["match_id", *CONTEXT_COLUMNS]
    assert np.isclose(out.loc[("top", 0), "top_priority"], 0.5)
    assert np.isclose(out.loc[("top", 0), "mid_priority"], 0.2)
    assert np.isclose(out.loc[("top", 0), "bot_priority"], 0.7)
    assert np.isclose(out.loc[("top", 0), "lane_priority"], 0.5)
    assert np.isclose(out.loc[("sup", 0), "lane_priority"], 0.7)
    assert np.isclose(out.loc[("jng", 0), "lane_priority"], 0.7)
    assert np.isclose(out.loc[("jng", 1), "top_priority"], 0.9)
    assert np.isclose(out.loc[("jng", 1), "mid_priority"], 0.9)
