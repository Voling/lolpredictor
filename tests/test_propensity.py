import numpy as np
import pandas as pd
import pytest

from synergy.features.propensity import (
    COVARIATES,
    KINDS,
    PROPENSITY_COLUMNS,
    _gamma_prior,
    fit_propensities,
    opportunity_rows,
)
from tests.test_features import BLUE_JUNGLE, MID, RED_JUNGLE, build_match


def timeline_with_tracks(tracks: dict[int, list[tuple[int, int]]], minutes: int = 16) -> dict:
    frames = []
    for minute in range(minutes):
        participant_frames = {}
        for index in range(10):
            pid = index + 1
            track = tracks.get(pid)
            position = track[min(minute, len(track) - 1)] if track else MID
            participant_frames[str(pid)] = {
                "participantId": pid,
                "totalGold": 500 * minute,
                "xp": 300 * minute,
                "minionsKilled": 6 * minute,
                "jungleMinionsKilled": 0,
                "level": 1 + minute // 2,
                "position": {"x": position[0], "y": position[1]},
                "damageStats": {"totalDamageDoneToChampions": 0, "totalDamageTaken": 0},
            }
        frames.append({"timestamp": minute * 60000, "participantFrames": participant_frames, "events": []})
    return {
        "metadata": {"matchId": "NA1_1", "participants": [f"p{i}" for i in range(10)]},
        "info": {
            "frameInterval": 60000,
            "participants": [{"participantId": i + 1, "puuid": f"p{i}"} for i in range(10)],
            "frames": frames,
        },
    }


def test_initiator_and_follower_are_separated():
    tracks = {
        2: [MID] * 5 + [RED_JUNGLE] * 11,
        3: [MID] * 6 + [RED_JUNGLE] * 10,
    }
    rows = opportunity_rows(build_match(), timeline_with_tracks(tracks))
    frame = pd.DataFrame(rows)
    initiate = frame[(frame["kind"] == "initiate") & (frame["outcome"] == 1.0)]
    follow = frame[(frame["kind"] == "follow") & (frame["outcome"] == 1.0)]
    assert set(initiate["puuid"]) == {"p1"}
    assert set(follow["puuid"]) == {"p2"}


def test_a_follow_opportunity_only_exists_once_an_invade_happened():
    quiet = opportunity_rows(build_match(), timeline_with_tracks({}))
    assert not [row for row in quiet if row["kind"] == "follow"]


def test_defend_opportunities_appear_when_an_enemy_enters_our_jungle():
    tracks = {7: [MID] * 4 + [BLUE_JUNGLE] * 12}
    frame = pd.DataFrame(opportunity_rows(build_match(), timeline_with_tracks(tracks)))
    defend = frame[frame["kind"] == "defend"]
    assert len(defend) > 0
    assert set(defend["puuid"]).issubset({f"p{i}" for i in range(5)})


def test_situation_covariates_are_lagged_behind_the_decision():
    tracks = {2: [MID] * 5 + [RED_JUNGLE] * 11}
    frame = pd.DataFrame(opportunity_rows(build_match(), timeline_with_tracks(tracks)))
    entry = frame[(frame["kind"] == "initiate") & (frame["puuid"] == "p1") & (frame["outcome"] == 1.0)]
    assert len(entry) == 1
    assert float(entry.iloc[0]["distance_to_enemy_jungle"]) == 1.0


def test_gamma_prior_shrinks_hard_when_players_look_alike():
    rng = np.random.default_rng(0)
    chances, rate = 50, 0.1
    expected = np.full(400, chances * rate)
    variance = np.full(400, chances * rate * (1 - rate))
    identical = rng.binomial(chances, rate, size=400).astype(float)
    skill = np.clip(rng.gamma(2.0, 0.5, size=400), 0.05, 5.0)
    varied = rng.binomial(chances, np.clip(rate * skill, 0, 1)).astype(float)
    assert _gamma_prior(identical, expected, variance) > _gamma_prior(varied, expected, variance)


def test_gamma_prior_uses_binomial_not_poisson_noise():
    rng = np.random.default_rng(1)
    chances, rate = 40, 0.5
    expected = np.full(500, chances * rate)
    binomial = np.full(500, chances * rate * (1 - rate))
    poisson = expected
    draws = rng.binomial(chances, rate, size=500).astype(float)
    assert _gamma_prior(draws, expected, poisson) >= _gamma_prior(draws, expected, binomial)


def test_propensities_are_zero_without_usable_opportunities():
    empty = pd.DataFrame(columns=["kind", "outcome", "puuid", "role", *COVARIATES])
    table, diagnostics = fit_propensities(empty)
    assert diagnostics == {}
    assert list(table.columns) == PROPENSITY_COLUMNS


def test_propensity_recovers_a_planted_tendency():
    rng = np.random.default_rng(7)
    rows = []
    players = [(f"eager{i}", 0.45) for i in range(20)] + [(f"shy{i}", 0.05) for i in range(20)]
    for player, rate in players:
        for _ in range(60):
            rows.append(
                {
                    "puuid": player,
                    "kind": "follow",
                    "role": "MIDDLE",
                    "minute": float(rng.integers(1, 15)),
                    "gold_diff": float(rng.normal()),
                    "is_jungler": 0.0,
                    "in_own_half": 1.0,
                    "distance_to_enemy_jungle": 1.0,
                    "unspent_gold": 0.4,
                    "outcome": float(rng.random() < rate),
                }
            )
    table, diagnostics = fit_propensities(pd.DataFrame(rows))
    eager = table.loc[[f"eager{i}" for i in range(20)], "prop_follow"].mean()
    shy = table.loc[[f"shy{i}" for i in range(20)], "prop_follow"].mean()
    assert eager > shy
    assert diagnostics["follow"]["player_dispersion"] > 0.01
    assert set(KINDS) >= set(diagnostics)


def test_propensity_is_conditional_not_a_raw_rate():
    rng = np.random.default_rng(3)
    rows = []
    for player, minutes in (("early", range(1, 5)), ("late", range(11, 15))):
        for _ in range(600):
            minute = float(rng.choice(list(minutes)))
            rows.append(
                {
                    "puuid": player,
                    "kind": "follow",
                    "role": "MIDDLE",
                    "minute": minute,
                    "gold_diff": 0.0,
                    "is_jungler": 0.0,
                    "in_own_half": 1.0,
                    "distance_to_enemy_jungle": 1.0,
                    "unspent_gold": 0.4,
                    "outcome": float(rng.random() < minute / 30.0),
                }
            )
    table, _ = fit_propensities(pd.DataFrame(rows))
    raw = pd.DataFrame(rows).groupby("puuid")["outcome"].mean()
    assert raw["late"] > raw["early"] * 1.5
    assert abs(table.loc["late", "prop_follow"] - table.loc["early", "prop_follow"]) < 0.4
