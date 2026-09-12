import pytest

from synergy.features.anchors import MAX_REACH, MAX_SPEED, plausible

PATH = [(0.0, 0.0, 0.0), (1.0, 1000.0, 1000.0), (2.0, 2000.0, 2000.0), (3.0, 3000.0, 3000.0)]


def test_a_spot_on_the_path_is_plausible():
    assert plausible(PATH, 1.5, (1500.0, 1500.0))


def test_a_spot_the_player_could_not_reach_is_rejected():
    assert not plausible(PATH, 1.5, (14000.0, 14000.0))


@pytest.mark.parametrize("minute", [-1.0, 99.0])
def test_a_spot_outside_the_known_positions_is_rejected(minute):
    assert not plausible(PATH, minute, (1500.0, 1500.0))


def test_reach_is_capped_even_at_the_midpoint_between_frames():
    assert MAX_SPEED * 30.0 > MAX_REACH
    assert plausible(PATH, 1.5, (1000.0 + MAX_REACH * 0.5, 1000.0))
    assert not plausible(PATH, 1.5, (1000.0 + MAX_REACH * 1.5, 1000.0))


def test_a_spot_right_next_to_a_known_position_must_be_close_to_it():
    assert not plausible(PATH, 1.02, (9000.0, 1000.0))


def test_objective_and_building_assists_are_claimed_too():
    from synergy.features.anchors import _claims

    monster = {"type": "ELITE_MONSTER_KILL", "killerId": 3, "assistingParticipantIds": [1, 2]}
    building = {"type": "BUILDING_KILL", "killerId": 7, "assistingParticipantIds": [8]}
    assert [pid for _, pid in _claims(monster)] == [3, 1, 2]
    assert [pid for _, pid in _claims(building)] == [7, 8]


def test_shop_events_claim_the_buyer_and_kills_claim_the_victim():
    from synergy.features.anchors import _claims

    assert _claims({"type": "ITEM_PURCHASED", "participantId": 4}) == [("shop", 4)]
    kill = {"type": "CHAMPION_KILL", "victimId": 2, "killerId": 5, "assistingParticipantIds": [6]}
    assert _claims(kill)[0] == ("certain", 2)


def test_the_death_timer_follows_the_level_table():
    from synergy.features.anchors import respawn_delay

    assert respawn_delay(1) == pytest.approx(10.0 / 60.0)
    assert respawn_delay(7) == pytest.approx(20.0 / 60.0)
    assert respawn_delay(18) == pytest.approx(52.5 / 60.0)
    assert respawn_delay(99) == respawn_delay(18)
    assert respawn_delay(0) == respawn_delay(1)


def test_a_player_is_dead_between_the_kill_and_the_respawn():
    from synergy.features.anchors import dead_at

    windows = [(7.0, 7.25), (11.0, 11.5)]
    assert dead_at(windows, 7.1)
    assert not dead_at(windows, 7.3)
    assert dead_at(windows, 11.0)
    assert not dead_at(windows, 11.5)


def test_a_claim_right_after_respawn_is_judged_from_the_fountain():
    from synergy.features.anchors import plausible

    certainties = [(7.0, 6000.0, 6000.0), (7.2, 400.0, 460.0), (8.0, 2000.0, 2000.0)]
    assert plausible(certainties, 7.3, (1200.0, 1200.0))
    assert not plausible(certainties, 7.3, (13000.0, 13000.0))
