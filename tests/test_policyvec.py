import numpy as np
import pandas as pd
import pytest

from synergy.config import Settings
from synergy.features.policy import ACTIONS, STATES
from synergy.features.policyvec import (
    MIN_MINUTES,
    fit_policy_vectors,
    load_policy_vectors,
    neighbours,
    standardise,
)


def build_policy(players: int = 400, minutes: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bias = {f"p{index}": rng.normal(0, 1) for index in range(players)}
    rows = []
    for puuid, tilt in bias.items():
        for minute in range(minutes):
            state = "even" if minute % 2 else STATES[minute % len(STATES)]
            if state == "even":
                roams = 1 / (1 + np.exp(-tilt))
                action = "roam" if rng.random() < roams else "shove"
            else:
                action = ACTIONS[rng.integers(0, len(ACTIONS))]
            rows.append({"match_id": f"M{minute}", "puuid": puuid, "state": state, "action": action})
    return pd.DataFrame(rows)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


def test_only_reliable_cells_survive(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    assert report["cells_kept"] < report["cells_tested"]
    assert set(vectors.columns) == set(report["reliability"])
    assert all(value >= report["min_reliability"] for value in report["reliability"].values())


def test_the_planted_cell_is_kept(settings):
    _, report = fit_policy_vectors(build_policy(), settings)
    assert "even|roam" in report["reliability"]


def test_vectors_round_trip(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    loaded, meta = load_policy_vectors(settings)
    assert meta == report
    pd.testing.assert_frame_equal(loaded[list(vectors.columns)], vectors, check_names=False)


def test_standardise_returns_unit_rows(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    norms = np.linalg.norm(standardise(vectors, report), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_players_below_the_minute_floor_are_dropped(settings):
    policy = build_policy(players=300, minutes=MIN_MINUTES + 10)
    thin = policy[policy.puuid == "p0"].head(MIN_MINUTES - 5)
    policy = pd.concat([policy[policy.puuid != "p0"], thin])
    vectors, _ = fit_policy_vectors(policy, settings)
    assert "p0" not in vectors.index


def test_neighbours_exclude_the_query_and_respect_role(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    roles = pd.Series(
        ["TOP" if index % 2 else "JUNGLE" for index in range(len(vectors))], index=vectors.index
    )
    found = neighbours("p0", vectors, report, limit=5, roles=roles)
    assert "p0" not in found.index
    assert (roles.reindex(found.index) == roles.loc["p0"]).all()
    assert found.is_monotonic_decreasing


def test_an_unknown_player_yields_nothing(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    assert neighbours("nobody", vectors, report).empty


def test_a_corpus_too_small_to_measure_keeps_no_cells(settings):
    vectors, report = fit_policy_vectors(build_policy(players=20, minutes=10), settings)
    assert vectors.empty
    assert report == {}
