import numpy as np
import pandas as pd

from synergy.features.reaction import (
    REACTION_COLUMNS,
    jungle_counts,
    objective_counts,
    response_counts,
    ward_counts,
)


def test_responses_map_to_one_situation_and_one_ordered_outcome_each():
    # given
    rows = pd.DataFrame(
        {
            "match_id": ["m"] * 4, "puuid": ["p"] * 4, "trigger": ["kill", "kill", "plate", "kill"],
            "ours": [1, 0, 1, 1], "is_actor": [0, 0, 0, 1], "is_victim": [0, 0, 0, 0],
            "approach": [1.0, 4.0, 9.0, 1.0], "present": [1, 1, 0, 1], "converged": [1, 0, 0, 1],
            "left_after": [0, 1, 0, 0], "held_ground": [0, 0, 0, 0],
        }
    )

    # when
    counts = response_counts(rows)

    # then
    assert len(counts) == 3
    assert counts.situation.tolist() == ["kill_ours_near", "kill_theirs_mid", "plate_ours_far"]
    assert counts.outcome.tolist() == ["converged", "left", "absent"]


def test_an_unknown_distance_counts_as_far_and_any_side_but_ours_as_theirs():
    # given
    rows = pd.DataFrame(
        {
            "match_id": ["m"] * 3, "puuid": ["p", "q", "r"], "trigger": ["building", "ward", "objective"],
            "ours": [2, 1, np.nan], "is_actor": [0, 0, 0], "is_victim": [0, 0, 0],
            "approach": [np.nan, 1.0, 2.5], "present": [0.0, 1.0, 1.0], "converged": [0.0, 0.0, 0.0],
            "left_after": [0.0, 0.0, 0.0], "held_ground": [0.0, 0.0, 1.0],
        }
    )

    # when
    counts = response_counts(rows)

    # then
    assert counts.puuid.tolist() == ["p", "r"]
    assert counts.situation.tolist() == ["building_theirs_far", "objective_theirs_mid"]
    assert counts.outcome.tolist() == ["absent", "held"]
    assert counts.dtypes.astype(str).tolist() == ["object", "object", "object", "object", "float64"]


def test_objective_ward_and_jungle_rows_land_in_declared_cells():
    # given
    objectives = pd.DataFrame({"match_id": ["m"], "puuid": ["p"], "objective": ["DRAGON"], "ours": [0], "o_approach_distance": [3.0],
                               "o_died": [0], "o_fought": [1], "o_committed": [1], "o_rotated_in": [0], "o_approaching": [1]})
    wards = pd.DataFrame({"match_id": ["m", "m"], "puuid": ["p", "p"], "minute": [2.0, 12.0], "zone": ["JUNGLE_ENEMY_TOPSIDE", "LANE_BOT_NEUTRAL"]})
    openings = pd.DataFrame({"match_id": ["m"], "puuid": ["p"], "first_gank_minute": [4.0], "first_invade_minute": [np.nan], "crossed_sides": [True]})

    # when
    obj, ward, jgl = objective_counts(objectives), ward_counts(wards), jungle_counts(openings)

    # then
    assert obj.situation.tolist() == ["DRAGON_theirs_mid"] and obj.outcome.tolist() == ["fought"]
    assert ward.situation.tolist() == ["early", "late"] and ward.outcome.tolist() == ["enemy_jungle", "lane_middle"]
    assert set(zip(jgl.situation, jgl.outcome)) == {("gank", "by5"), ("invade", "never"), ("sides", "crossed")}
    for frame in (obj, ward, jgl):
        prefix = {"DRAGON_theirs_mid": "obj", "early": "ward", "gank": "jgl"}.get(frame.situation.iloc[0], "jgl")
        for s, o in zip(frame.situation, frame.outcome):
            assert f"{prefix}_{s}_{o}" in REACTION_COLUMNS
