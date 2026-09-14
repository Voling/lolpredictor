import pytest

from synergy.config import Settings
from synergy.features.build import qualifying_matches
from synergy.ranks import rank_to_lp

MASTER = rank_to_lp("MASTER", None, 0)


class FakeStore:
    def __init__(self, matches: dict[str, list[int | None]], patch: str | None = None):
        self.matches = matches
        self.patch = patch or f"{Settings().season}.4"

    def match_rank_summary(self) -> list[dict]:
        rows = []
        for match_id, values in self.matches.items():
            known = [lp for lp in values if lp is not None]
            rows.append(
                {
                    "match_id": match_id,
                    "patch": self.patch,
                    "ranked": len(known),
                    "average_lp": sum(known) / len(known) if known else None,
                }
            )
        return rows


def settings(floor: int = MASTER, ranked: int = 2) -> Settings:
    return Settings(min_average_lp=floor, min_ranked_participants=ranked)


@pytest.mark.parametrize(
    "lps, kept",
    [
        ([2800, 3000, 3200, 2900], True),
        ([2400, 2500, 2600, 2450], False),
        ([2800, 2800], True),
        ([2400, 3400], True),
        ([3200, 3400, None, None], True),
        ([2000, None, None, None], True),
    ],
    ids=["master", "diamond", "on the floor", "average not worst", "unranked ignored", "unjudgeable"],
)
def test_the_floor_is_applied_to_the_average_of_known_ranks(lps, kept):
    found = qualifying_matches(FakeStore({"m1": lps}), settings())
    assert found == ({"m1"} if kept else set())


def test_raising_the_evidence_bar_keeps_matches_it_can_no_longer_judge():
    store = FakeStore({"m1": [2000, 2100, None, None]})
    assert qualifying_matches(store, settings(ranked=2)) == set()
    assert qualifying_matches(store, settings(ranked=3)) == {"m1"}


@pytest.mark.parametrize(
    "store, config",
    [
        (FakeStore({"m1": [100, 200]}), settings(floor=0)),
        (FakeStore({"m1": [None, None]}), settings()),
    ],
    ids=["zero floor", "nothing ranked"],
)
def test_the_rank_floor_disables_itself_when_it_cannot_decide(store, config):
    assert qualifying_matches(store, config) == {"m1"}


def test_the_season_rule_still_applies_when_the_rank_floor_cannot_decide():
    assert qualifying_matches(FakeStore({"m1": [None, None]}, patch="15.24"), settings()) == set()


def test_mixed_corpus_keeps_only_the_qualifying_matches():
    store = FakeStore(
        {"high": [3000, 3100, 3200], "low": [2400, 2300, 2500], "edge": [2800, 2800, 2800]}
    )
    assert qualifying_matches(store, settings()) == {"high", "edge"}


def test_matches_from_another_season_are_dropped_whatever_their_rank():
    lineup = {"m1": [MASTER] * 10}
    inside = qualifying_matches(FakeStore(lineup), settings())
    outside = qualifying_matches(FakeStore(lineup, patch="15.24"), settings())
    assert inside == {"m1"}
    assert outside == set()


def test_the_season_rule_needs_the_patch_column():
    class Patchless(FakeStore):
        def match_rank_summary(self):
            return [
                {key: value for key, value in row.items() if key != "patch"}
                for row in super().match_rank_summary()
            ]

    with pytest.raises(ValueError, match="patch"):
        qualifying_matches(Patchless({"m1": [MASTER] * 10}), settings())
