import pytest

from synergy.features.positions import POSITIONS, combination, parse_position


def test_positions_parse_from_the_names_players_use():
    # given
    spoken = ["top", "JG", "Jungle", "mid", "adc", "bot", "supp", "Support"]

    # when
    parsed = [parse_position(text) for text in spoken]

    # then
    assert parsed == ["TOP", "JUNGLE", "JUNGLE", "MIDDLE", "BOTTOM", "BOTTOM", "UTILITY", "UTILITY"]
    assert set(parsed) <= set(POSITIONS)


def test_an_unknown_position_is_refused_with_the_accepted_names():
    # given
    text = "carry"

    # when
    with pytest.raises(ValueError) as refusal:
        parse_position(text)

    # then
    assert "carry" in str(refusal.value) and "support" in str(refusal.value)


def test_a_position_pair_is_the_same_key_in_either_order():
    # given
    left, right = "TOP", "JUNGLE"

    # when
    forward, backward = combination(left, right), combination(right, left)

    # then
    assert forward == backward == "JUNGLE+TOP"
