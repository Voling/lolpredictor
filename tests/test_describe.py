from synergy.features.describe import phrase, describe, describe_situation, named, situation_of
from synergy.features.priority import PRIORITY_COLUMNS
from synergy.features.reaction import REACTION_COLUMNS
from synergy.features.tendency import TENDENCY_COLUMNS


def test_every_named_cell_is_described_in_words():
    # given
    cells = [*REACTION_COLUMNS, *PRIORITY_COLUMNS, *TENDENCY_COLUMNS]

    # when
    words = [describe(cell) for cell in cells]

    # then
    assert all(word != cell and "_" not in word for word, cell in zip(words, cells))
    assert describe("rsp_kill_ours_far_converged") == "converged on an ally kill from far away"
    assert describe("obj_DRAGON_theirs_far_rotated") == "rotated toward an enemy dragon from far away"
    assert describe("ward_early_lane_middle") == "ward placed before 5 minutes in the middle of the lane"
    assert describe("jgl_gank_by5") == "first gank by 5 minutes"
    assert describe("prio_pre_objective_deep_own_absent_farm") == (
        "standing deep in own half, opponent away, farming in the minute before an objective"
    )
    assert describe("tend_greed_punished_t--_own") == "punished for greed with top holding priority, on own half"
    assert describe("tend_initiate_---_away") == "starts a fight with no lane holding priority, on the enemy half"


def test_components_are_labelled_but_not_named():
    # given
    cells = ["habit_3", "move_0", "style_e7"]

    # when
    words = [describe(cell) for cell in cells]

    # then
    assert words == ["habit component 3", "movement component 0", "embedding component 7"]
    assert not any(named(cell) for cell in cells) and named("rsp_kill_ours_far_converged")


def test_cells_group_into_situations_with_their_own_words():
    # given
    cells = ["rsp_kill_ours_far_converged", "rsp_kill_ours_far_absent", "ward_late_river", "prio_all_dead", "tend_dive_tmb_own"]

    # when
    situations = [situation_of(cell) for cell in cells]

    # then
    assert situations == ["rsp_kill_ours_far", "rsp_kill_ours_far", "ward_late", "prio_all", "tend_dive"]
    assert describe_situation("rsp_kill_ours_far") == "after an ally kill from far away"
    assert describe_situation("tend_dive") == "diving"
    assert describe_situation("prio_pre_objective") == "lane state in the minute before an objective"


def test_every_named_cell_reads_as_a_present_tense_habit():
    # given
    from synergy.features.describe import named

    cells = [
        "rsp_kill_ours_far_converged",
        "obj_DRAGON_theirs_far_rotated",
        "ward_early_lane_middle",
        "jgl_gank_by5",
        "jgl_sides_crossed",
        "prio_all_own_absent_farm",
        "tend_greed_punished_tm-_away",
        "tend_dive_tmb_own",
    ]

    # when
    phrases = [phrase(cell) for cell in cells]

    # then
    assert phrases == [
        "converges on an ally kill from far away",
        "rotates toward an enemy dragon from far away",
        "places wards before 5 minutes in the middle of the lane",
        "ganks first by 5 minutes",
        "crosses to the other side of the jungle",
        "is standing on own side, opponent away, farming",
        "gets punished for greed with top and mid holding priority, on the enemy half",
        "dives with top, mid and bot holding priority, on own half",
    ]
    assert all(named(cell) for cell in cells) and phrase("habit_3") == "habit component 3"
