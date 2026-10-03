import numpy as np
import pandas as pd

from synergy.features.cells import SeatIndex, dense_counts
from synergy.features.events import event_response_rows
from synergy.features.reaction import (
    REACTION_COLUMNS,
    STATES,
    chance_state,
    jungle_counts,
    objective_counts,
    response_counts,
    ward_counts,
)
from tests.test_features import build_match, build_timeline


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
    assert counts.state.tolist() == ["early_even", "early_even"]
    assert counts.dtypes.astype(str).tolist() == ["object", "object", "object", "object", "object", "float64"]


def test_a_chance_is_early_before_ten_minutes_and_behind_or_ahead_past_five_hundred_gold():
    # given
    chances = [(9.9, -501.0), (10.0, 501.0), (3.0, 500.0), (12.5, -500.0), (0.2, 0.0)]

    # when
    states = [chance_state(minute, lead) for minute, lead in chances]

    # then
    assert states == ["early_behind", "late_ahead", "early_even", "late_even", "early_even"]
    assert set(states) <= set(STATES) and len(STATES) == 6


def test_a_reaction_reads_the_lane_gold_against_the_same_position_a_minute_before_the_trigger():
    # given
    match, timeline = build_match(), build_timeline(events=[
        {"type": "CHAMPION_KILL", "timestamp": int(12.5 * 60000), "killerId": 3, "victimId": 8, "assistingParticipantIds": [], "position": {"x": 7500, "y": 7500}},
    ])
    frames = timeline["info"]["frames"]
    frames[11]["participantFrames"]["1"]["totalGold"] += 800
    frames[12]["participantFrames"]["1"]["totalGold"] -= 2000
    frames[11]["participantFrames"]["5"]["totalGold"] += 900
    match["info"]["participants"][9]["teamPosition"] = ""

    # when
    rows = event_response_rows(match, timeline)

    # then
    state = {row["puuid"]: row["state"] for row in rows}
    assert state["p0"] == "late_ahead" and state["p5"] == "late_behind"
    assert state["p4"] == "late_even" and state["p1"] == "late_even"


def test_counts_carry_the_state_of_each_chance_into_their_own_slot():
    # given
    rows = pd.DataFrame(
        {
            "match_id": ["m", "m"], "puuid": ["p", "p"], "trigger": ["kill", "kill"], "ours": [1, 1], "is_actor": [0, 0], "is_victim": [0, 0],
            "approach": [1.0, 1.0], "present": [1, 1], "converged": [1, 0], "left_after": [0, 0], "held_ground": [0, 1],
            "state": ["early_behind", "late_ahead"],
        }
    )
    objectives = pd.DataFrame({"match_id": ["m"], "puuid": ["p"], "objective": ["DRAGON"], "side": ["ours"], "o_approach_distance": [9.0], "state": ["late_behind"],
                               "o_died": [0], "o_fought": [0], "o_committed": [0], "o_rotated_in": [0], "o_approaching": [0]})
    index = SeatIndex(pd.DataFrame({"match_id": ["m"], "puuid": ["p"], "position": ["TOP"]}))

    # when
    counts = response_counts(rows)
    dense = dense_counts(counts, index, ["kill_ours_near"], ["converged", "held"], states=STATES)

    # then
    assert counts.state.tolist() == ["early_behind", "late_ahead"] and objective_counts(objectives).state.tolist() == ["late_behind"]
    assert dense[0, 0, STATES.index("early_behind")].tolist() == [1.0, 0.0] and dense[0, 0, STATES.index("late_ahead")].tolist() == [0.0, 1.0]
    assert dense.sum() == 2.0


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


def test_gold_readings_cover_kill_and_plate_reactions_when_behind_and_when_ahead():
    # given
    from synergy.features.reaction import GOLD_COLUMNS, GOLD_SPLIT

    # when
    parts = [column.split("_") for column in GOLD_COLUMNS]

    # then
    assert len(GOLD_COLUMNS) == 120 and set(GOLD_COLUMNS) <= set(REACTION_COLUMNS) and len(REACTION_COLUMNS) == 435
    assert {part[1] for part in parts} == {"behind", "ahead"} and {part[2] for part in parts} == {"kill", "plate"} and {part[0] for part in parts} == {"rspg"}
    assert GOLD_SPLIT["groups"] == {"behind": ["early_behind", "late_behind"], "ahead": ["early_ahead", "late_ahead"]}
