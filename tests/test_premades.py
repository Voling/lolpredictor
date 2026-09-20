import pandas as pd

from synergy.ingest.premades import seed_order, sessions, shared_games


def _frame():
    when = pd.to_datetime(
        ["2026-03-01 20:00", "2026-03-01 20:40", "2026-03-01 21:20", "2026-06-01 10:00", "2026-06-02 10:00"], utc=True
    )
    return pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4", "m5", "m5"],
            "puuid": ["a", "b", "a", "b", "a", "b", "a", "c", "a", "c"],
            "team_id": [100] * 10,
            "game_creation": [when[0], when[0], when[1], when[1], when[2], when[2], when[3], when[3], when[4], when[4]],
        }
    )


def test_shared_games_count_every_shared_match_and_the_ones_in_close_succession():
    # given
    frame = _frame()

    # when
    table = shared_games(frame).set_index(["left", "right"])

    # then
    assert table.loc[("a", "b"), "shared"] == 3 and table.loc[("a", "b"), "close"] == 2
    assert table.loc[("a", "c"), "shared"] == 2 and table.loc[("a", "c"), "close"] == 0
    assert sessions(frame) == {("a", "b")}


def test_seed_order_keeps_master_and_above_pairs_and_puts_the_strongest_duo_first():
    # given
    pairs = pd.DataFrame(
        {"left": ["a", "a", "d"], "right": ["b", "c", "e"], "shared": [3, 9, 5], "close": [2, 4, 3]}
    )
    tiers = pd.Series({"a": "MASTER", "b": "MASTER", "c": "DIAMOND", "d": "CHALLENGER", "e": "GRANDMASTER"})
    games = pd.Series({"a": 40, "b": 12, "d": 300, "e": 80})

    # when
    order = seed_order(pairs, tiers, games)

    # then
    assert order.index.tolist() == ["e", "d", "b", "a"]
    assert "c" not in order.index
    assert order.loc["a", "strongest_pair"] == 3 and order.loc["e", "games"] == 80
