from synergy.config import Settings
from synergy.features.build import qualifying_matches
from synergy.ranks import rank_to_lp

MASTER = rank_to_lp("MASTER", None, 0)


class FakeStore:
    def __init__(self, matches: dict[str, list[int | None]]):
        self.matches = matches

    def match_rank_summary(self) -> list[dict]:
        rows = []
        for match_id, values in self.matches.items():
            known = [lp for lp in values if lp is not None]
            rows.append(
                {
                    "match_id": match_id,
                    "ranked": len(known),
                    "average_lp": sum(known) / len(known) if known else None,
                }
            )
        return rows


def settings(floor: int = MASTER, ranked: int = 2) -> Settings:
    return Settings(min_average_lp=floor, min_ranked_participants=ranked)


def test_master_zero_lp_is_the_documented_floor():
    assert MASTER == 2800


def test_a_master_lobby_is_kept():
    store = FakeStore({"m1": [2800, 3000, 3200, 2900]})
    assert qualifying_matches(store, settings()) == {"m1"}


def test_a_diamond_lobby_is_dropped():
    store = FakeStore({"m1": [2400, 2500, 2600, 2450]})
    assert qualifying_matches(store, settings()) == set()


def test_the_floor_is_inclusive():
    store = FakeStore({"m1": [2800, 2800]})
    assert qualifying_matches(store, settings()) == {"m1"}


def test_average_decides_not_the_worst_player():
    store = FakeStore({"m1": [2400, 3400]})
    assert qualifying_matches(store, settings()) == {"m1"}


def test_a_match_with_too_few_known_ranks_is_kept_rather_than_guessed():
    store = FakeStore({"m1": [2000, None, None, None]})
    assert qualifying_matches(store, settings(ranked=2)) == {"m1"}


def test_raising_the_evidence_bar_keeps_more_matches():
    store = FakeStore({"m1": [2000, 2100, None, None]})
    assert qualifying_matches(store, settings(ranked=2)) == set()
    assert qualifying_matches(store, settings(ranked=3)) == {"m1"}


def test_unranked_participants_do_not_drag_the_average_down():
    store = FakeStore({"m1": [3200, 3400, None, None]})
    assert qualifying_matches(store, settings()) == {"m1"}


def test_a_zero_floor_disables_the_filter():
    store = FakeStore({"m1": [100, 200]})
    assert qualifying_matches(store, settings(floor=0)) is None


def test_no_ranked_players_at_all_disables_the_filter():
    store = FakeStore({"m1": [None, None]})
    assert qualifying_matches(store, settings()) is None


def test_mixed_corpus_keeps_only_the_qualifying_matches():
    store = FakeStore(
        {
            "high": [3000, 3100, 3200],
            "low": [2400, 2300, 2500],
            "edge": [2800, 2800, 2800],
        }
    )
    assert qualifying_matches(store, settings()) == {"high", "edge"}
