import numpy as np
import pandas as pd

from synergy.features.positions import POSITIONS
from synergy.ml.duos import History, duo_records, halves, held_out, keyed, served_totals, shrunk, signed_residuals, split_half


def _games(duos: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    rows = []
    for (left, right), residuals in duos.items():
        for index, residual in enumerate(residuals):
            first, second = (left, right) if index % 2 == 0 else (right, left)
            rows.append({"match_id": f"{left}{right}{index:03d}", "left": first, "right": second, "residual": residual})
    return pd.DataFrame(rows)


def test_a_duo_keeps_more_of_its_record_the_more_games_it_has_and_its_later_games_set_the_spread():
    # given
    duos = {(f"p{index}", f"q{index}"): [mean + 1000.0, mean - 1000.0] * 5 for index, mean in enumerate([-600.0, -400.0, -200.0, 200.0, 400.0, 600.0] * 2)}
    duos[("short", "pair")] = [1600.0, -400.0]
    duos[("once", "only")] = [900.0]

    # when
    pairs = keyed(_games(duos))
    kept, spread, noise = duo_records(pairs, *halves(pairs))
    record = kept.set_index(["a", "b"])

    # then
    assert spread > 0.0 and noise > 1000.0
    assert record.loc[("p5", "q5"), "games"] == 10
    assert 0.0 < record.loc[("p5", "q5"), "record"] < 600.0
    assert -600.0 < record.loc[("p0", "q0"), "record"] < 0.0
    assert 0.0 < record.loc[("pair", "short"), "record"] < record.loc[("p5", "q5"), "record"]
    assert ("once", "only") not in record.index


def test_duos_that_all_share_one_mean_have_no_spread_and_no_record():
    # given
    duos = {(f"p{index}", f"q{index}"): [1000.0, -1000.0] * 5 for index in range(8)}

    # when
    pairs = keyed(_games(duos))
    kept, spread, _ = duo_records(pairs, *halves(pairs))

    # then
    assert spread == 0.0
    assert (kept["record"] == 0.0).all()


def test_the_split_half_check_compares_each_duo_with_its_own_later_games():
    # given
    duos = {(f"p{index}", f"q{index}"): [mean] * 10 for index, mean in enumerate([-300.0, -100.0, 100.0, 300.0])}

    # when
    found = split_half(*halves(keyed(_games(duos))))

    # then
    assert found == {"duos": 4, "r": 1.0, "slope": 1.0}


def test_shrunk_group_means_keep_their_order_and_move_toward_zero():
    # given
    keys = np.array(["low"] * 8 + ["mid"] * 8 + ["high"] * 8 + ["lone"])
    values = np.concatenate([np.full(8, -300.0), np.zeros(8), np.full(8, 300.0), [900.0]])

    # when
    index, effect, games, tau = shrunk(keys, values, sigma2=100.0**2)
    low, mid, high, lone = (effect[index.get_loc(name)] for name in ("low", "mid", "high", "lone"))

    # then
    assert tau > 0.0
    assert -300.0 < low < mid < high < 300.0
    assert abs(low) > 250.0 and 0.0 < lone < 900.0 and games[index.get_loc("lone")] == 1


def _corpus(matches: int = 120, seed: int = 2):
    rng = np.random.default_rng(seed)
    positions = np.array([list(POSITIONS) * 2] * matches, dtype=object)
    names = np.array([[f"{position}{rng.integers(6)}" for position in row] for row in positions], dtype=object)
    champion = np.array(
        [[f"{position}_c{int(name[-1]) % 3 if rng.random() < 0.8 else rng.integers(3)}" for position, name in zip(row, seats)] for row, seats in zip(positions, names)],
        dtype=object,
    )
    blue = np.tile(np.arange(5), (matches, 1))
    red = np.tile(np.arange(5, 10), (matches, 1))
    planted_champion = np.where(np.char.endswith(champion.astype(str), "c0"), 400.0, 0.0)
    planted_form = np.where(np.char.endswith(names.astype(str), "1"), 300.0, 0.0)
    gold = 8000.0 + planted_champion + planted_form + rng.normal(0.0, 150.0, size=names.shape)
    return positions, names, champion, blue, red, gold


def test_history_recovers_planted_champion_and_player_effects_and_pools_them_for_serving():
    # given
    positions, names, champion, blue, red, gold = _corpus()
    rows = np.arange(len(gold))
    signed = signed_residuals(gold, np.zeros_like(gold), blue, red, rows)
    pools = pd.DataFrame({"puuid": names.ravel(), "position": positions.ravel(), "champion_name": champion.ravel()}).groupby(["puuid", "position", "champion_name"]).size().rename("games").reset_index()

    # when
    history = History(positions, names, champion, signed, rows, pools)
    table = history.table().set_index(["puuid", "position"])
    champions = history.champions().set_index(["position", "champion"])

    # then
    assert champions.loc[("TOP", "TOP_c0"), "effect"] > 250.0 > champions.loc[("TOP", "TOP_c1"), "effect"]
    assert table.loc[("TOP1", "TOP"), "form"] > 50.0 > table.loc[("TOP0", "TOP"), "form"]
    assert table.loc[("TOP0", "TOP"), "champion"] > table.loc[("TOP1", "TOP"), "champion"] and table["games"].sum() == gold.size
    assert set(table.columns) == {"games", "form", "champion"}


def test_held_out_scores_improve_with_champion_and_form_on_a_planted_corpus():
    # given
    positions, names, champion, blue, red, gold = _corpus(matches=200)
    rows = np.arange(len(gold))
    fit, test = rows[:150], rows[150:]
    pools = pd.DataFrame({"puuid": names[fit].ravel(), "position": positions[fit].ravel(), "champion_name": champion[fit].ravel()}).groupby(["puuid", "position", "champion_name"]).size().rename("games").reset_index()
    signed = signed_residuals(gold, np.zeros_like(gold), blue, red, rows)

    # when
    checked = held_out(gold, np.zeros_like(gold), blue, red, test, positions, names, champion, History(positions, names, champion, signed, fit, pools))

    # then
    assert set(checked) == set(POSITIONS)
    for found in checked.values():
        assert found["with_champion_played"] > found["readings"]
        assert found["with_form"] > found["with_champion_pool"] >= found["readings"]
        assert 0.3 < found["slope"] < 1.5


def test_the_score_distribution_adds_both_readings_the_fit_the_record_and_the_players_history_like_serving(serving_settings):
    # given
    games = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "left": ["a", "a"],
            "right": ["b", "nobody"],
            "left_position": ["TOP", "TOP"],
            "right_position": ["JUNGLE", "JUNGLE"],
            "residual": [0.0, 0.0],
        }
    )
    kept = pd.DataFrame({"a": ["a"], "b": ["b"], "games": [12], "mean": [100.0], "record": [40.0]})
    history = pd.DataFrame({"puuid": ["a"], "position": ["TOP"], "games": [20], "form": [30.0], "champion": [15.0]})

    # when
    plain, known, quantiles = served_totals(serving_settings, games, kept)
    with_history, _, _ = served_totals(serving_settings, games, kept, history)
    halved, _, halved_quantiles = served_totals(serving_settings, games, kept, history, np.array([0.5, 0.5, 1.0, 1.0, 1.0]))

    # then
    assert known.tolist() == [True, False]
    assert np.allclose(plain, [100.0 + 50.0 + 5.0 + 40.0])
    assert np.allclose(with_history, [100.0 + 30.0 + 15.0 + 50.0 + 5.0 + 40.0])
    assert quantiles.shape == (5, 1001) and quantiles[0, 0] == 100.0
    assert np.allclose(halved, [0.5 * (100.0 + 30.0 + 15.0) + 0.5 * 50.0 + 5.0 + 40.0])
    assert halved_quantiles[0, 0] == 0.5 * (100.0 + 30.0 + 15.0)
