import pandas as pd

from synergy.features.pools import ranked_pool, top_champions


def test_each_position_keeps_its_own_three_most_played_champions_newest_first_on_ties():
    # given
    rows = [
        *[("p", "TOP", "Pantheon", 100 + day) for day in range(5)],
        *[("p", "TOP", "Zaahen", 200 + day) for day in range(2)],
        ("p", "TOP", "Jax", 300),
        ("p", "TOP", "KSante", 50),
        *[("p", "BOTTOM", "Ezreal", 400 + day) for day in range(3)],
        ("p", "BOTTOM", "Caitlyn", 500),
        ("p", "BOTTOM", "Kaisa", 600),
        ("p", "BOTTOM", "Sivir", 450),
        ("p", "UNKNOWN", "Pantheon", 700),
        ("q", "MIDDLE", "Ahri", 800),
    ]
    seats = pd.DataFrame(rows, columns=["puuid", "position", "champion_name", "game_creation"])

    # when
    table = top_champions(seats, keep={"p"})

    # then
    found = {position: champions for position, champions in zip(table["position"], table["champions"])}
    assert found == {"BOTTOM": "Ezreal,Kaisa,Caitlyn", "TOP": "Pantheon,Zaahen,Jax"}
    assert set(table["puuid"]) == {"p"}


def test_an_evaluated_pool_is_ranked_by_games_and_keeps_the_newest_first_on_ties():
    # given
    champions = {"Jhin": 2, "Ezreal": 5, "Caitlyn": 2, "Kaisa": 1}

    # when
    pool = ranked_pool(champions)

    # then
    assert pool == ["Ezreal", "Jhin", "Caitlyn"]
