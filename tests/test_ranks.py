import pytest

from synergy.ranks import division_to_int, pick_solo_entry, rank_to_lp, tier_index


def test_division_ordering():
    assert division_to_int("IV") == 4
    assert division_to_int("I") == 1
    assert division_to_int(None) == 1


@pytest.mark.parametrize(
    "tier,division,lp,expected",
    [
        ("IRON", "IV", 0, 0),
        ("GOLD", "IV", 0, 1200),
        ("GOLD", "I", 50, 1550),
        ("DIAMOND", "II", 25, 2625),
        ("MASTER", None, 300, 3100),
        ("CHALLENGER", None, 1200, 4000),
    ],
)
def test_rank_to_lp(tier, division, lp, expected):
    assert rank_to_lp(tier, division, lp) == expected


def test_rank_to_lp_is_monotonic_within_a_tier():
    ladder = [rank_to_lp("PLATINUM", division, 0) for division in ("IV", "III", "II", "I")]
    assert ladder == sorted(ladder)


def test_unknown_tier_scores_zero():
    assert rank_to_lp("WOOD", "I", 40) == 0
    assert tier_index("WOOD") == -1


def test_tier_index_orders_the_ladder():
    assert tier_index("IRON") < tier_index("DIAMOND") < tier_index("CHALLENGER")


def test_pick_solo_entry_ignores_flex():
    entries = [
        {"queueType": "RANKED_FLEX_SR", "tier": "GOLD"},
        {"queueType": "RANKED_SOLO_5x5", "tier": "DIAMOND"},
    ]
    assert pick_solo_entry(entries)["tier"] == "DIAMOND"
    assert pick_solo_entry([]) is None
