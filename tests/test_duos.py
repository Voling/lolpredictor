import numpy as np
import pandas as pd

from synergy.ml.duos import duo_records, halves, keyed, served_totals, split_half


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


def test_the_score_distribution_adds_both_readings_the_fit_and_the_record_like_serving(serving_settings):
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

    # when
    total, known = served_totals(serving_settings, games, kept)

    # then
    assert known.tolist() == [True, False]
    assert np.allclose(total, [100.0 + 50.0 + 5.0 + 40.0])
