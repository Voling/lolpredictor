import pandas as pd

from synergy.features.events import one_per_attempt
from synergy.features.objectives import PITS, attempts, objective_rows, up_windows
from synergy.features.reaction import REACTION_COLUMNS, objective_counts
from tests.test_features import build_match, build_timeline


def _take(minute, team, kind):
    return {"minute": minute, "subtype": None, "killer_team": team, "x": PITS[kind][0], "y": PITS[kind][1]}


def _kill(minute, at, killer=7, victim=2):
    return {"type": "CHAMPION_KILL", "timestamp": int(round(minute * 60000)), "killerId": killer, "victimId": victim, "assistingParticipantIds": [], "position": {"x": at[0], "y": at[1]}}


def test_objectives_are_up_from_their_spawn_until_cleared_and_dragons_come_back():
    # given
    limit = 20.0

    # when
    dragon = up_windows("DRAGON", [7.0, 14.0], limit)
    grubs = up_windows("HORDE", [8.5, 8.6, 8.7], limit)
    two_grubs = up_windows("HORDE", [9.0, 9.1], limit)
    herald = up_windows("RIFTHERALD", [], 14.0)

    # then
    assert dragon == [(5.0, 7.0), (12.0, 14.0), (19.0, 20.0)]
    assert grubs == [(8.0, 8.7)] and two_grubs == [(8.0, 15.0)] and herald == []


def test_three_grubs_taken_together_are_one_attempt_won_by_the_team_with_most_of_them():
    # given
    pit = PITS["HORDE"]
    takes = [_take(8.50, 100, "HORDE"), _take(8.55, 200, "HORDE"), _take(8.60, 100, "HORDE")]
    kills = [_kill(8.45, pit), _kill(10.0, pit), _kill(16.0, pit)]

    # when
    found = attempts("HORDE", takes, kills, 20.0)

    # then
    assert [(attempt["start"], attempt["minute"], attempt["killer_team"]) for attempt in found] == [(8.45, 8.60, 100)]


def test_a_fight_at_a_live_dragon_is_an_attempt_even_without_a_take():
    # given
    pit = PITS["DRAGON"]
    takes = [_take(7.0, 200, "DRAGON")]
    kills = [_kill(5.5, pit), _kill(5.9, pit), _kill(6.2, (pit[0] - 6000.0, pit[1])), _kill(6.9, pit), _kill(7.5, pit)]

    # when
    found = attempts("DRAGON", takes, kills, 20.0)

    # then
    assert [(attempt["start"], attempt["minute"], attempt["killer_team"]) for attempt in found] == [(5.5, 5.9, None), (6.9, 7.0, 200)]
    assert (found[0]["x"], found[0]["y"]) == pit


def test_only_the_first_grub_of_each_attempt_triggers_a_reaction():
    # given
    rows = [
        {"kind": "objective", "detail": "HORDE", "minute": 8.50},
        {"kind": "objective", "detail": "HORDE", "minute": 8.55},
        {"kind": "kill", "detail": "", "minute": 8.52},
        {"kind": "objective", "detail": "DRAGON", "minute": 8.58},
        {"kind": "objective", "detail": "HORDE", "minute": 8.60},
        {"kind": "objective", "detail": "HORDE", "minute": 12.0},
    ]

    # when
    kept = one_per_attempt(rows)

    # then
    assert [(row["detail"], row["minute"]) for row in kept] == [("HORDE", 8.50), ("", 8.52), ("DRAGON", 8.58), ("HORDE", 12.0)]


def test_objective_rows_carry_one_attempt_per_grub_fight_and_mark_fights_with_no_take():
    # given
    grub, dragon = PITS["HORDE"], PITS["DRAGON"]
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": int(minute * 60000), "killerId": 2, "killerTeamId": 100, "monsterType": "HORDE", "position": {"x": grub[0], "y": grub[1]}}
        for minute in (8.50, 8.55, 8.60)
    ] + [_kill(6.0, dragon)]

    # when
    rows = objective_rows(build_match(), build_timeline(events=events))
    counts = objective_counts(pd.DataFrame(rows))

    # then
    grubs = [row for row in rows if row["objective"] == "HORDE"]
    fights = [row for row in rows if row["objective"] == "DRAGON"]
    assert len(grubs) == 10 and {row["side"] for row in grubs if row["team_id"] == 100} == {"ours"} and {row["side"] for row in grubs if row["team_id"] == 200} == {"theirs"}
    assert len(fights) == 10 and {row["side"] for row in fights} == {"none"} and not any(row["ours"] for row in fights)
    assert next(row for row in fights if row["puuid"] == "p1")["o_died"] == 1.0
    assert all(f"obj_{situation}_{outcome}" in REACTION_COLUMNS for situation, outcome in zip(counts.situation, counts.outcome))
    assert set(counts.situation.str.split("_").str[1]) == {"ours", "theirs", "none"}
