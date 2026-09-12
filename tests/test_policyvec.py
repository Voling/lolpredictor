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


def test_only_reliable_cells_survive_and_the_planted_one_does(settings):
    vectors, report = fit_policy_vectors(build_policy(), settings)
    assert report["cells_kept"] < report["cells_tested"]
    assert set(vectors.columns) == set(report["reliability"])
    assert all(value >= report["min_reliability"] for value in report["reliability"].values())
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
    assert neighbours("nobody", vectors, report).empty


def test_a_corpus_too_small_to_measure_keeps_no_cells(settings):
    vectors, report = fit_policy_vectors(build_policy(players=20, minutes=10), settings)
    assert vectors.empty
    assert report == {}


def test_weighting_pulls_the_vector_toward_the_more_reliable_cell():
    report = {
        "reliability": {"a": 0.9, "b": 0.1},
        "mean": {"a": 0.0, "b": 0.0},
        "std": {"a": 1.0, "b": 1.0},
    }
    vectors = pd.DataFrame({"a": [1.0], "b": [1.0]})
    plain = standardise(vectors, report, weighted=False)
    weighted = standardise(vectors, report, weighted=True)
    assert plain[0][0] == pytest.approx(plain[0][1])
    assert weighted[0][0] > weighted[0][1]
    assert np.linalg.norm(weighted[0]) == pytest.approx(1.0)


def test_equal_reliability_makes_weighting_a_no_op():
    report = {
        "reliability": {"a": 0.7, "b": 0.7},
        "mean": {"a": 0.0, "b": 0.0},
        "std": {"a": 1.0, "b": 1.0},
    }
    vectors = pd.DataFrame({"a": [1.0], "b": [-2.0]})
    assert np.allclose(
        standardise(vectors, report, weighted=False), standardise(vectors, report, weighted=True)
    )
