import numpy as np
import pandas as pd
import pytest

from synergy.features.conditional import player_lift, prior_table, separable
from synergy.features.matchup import counter_state, lane_pairs, matchup_edges, with_ally_state


def build_participations(matches: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strong, weak = "Darius", "Nasus"
    rows = []
    for index in range(matches):
        blue_top, red_top = (strong, weak) if index % 2 else (weak, strong)
        edge = 900.0 if blue_top == strong else -900.0
        for team, champion, sign in ((100, blue_top, 1.0), (200, red_top, -1.0)):
            rows.append(
                {
                    "match_id": f"NA1_{index}",
                    "team_id": team,
                    "puuid": f"top{team}_{index % 7}",
                    "position": "TOP",
                    "champion_name": champion,
                    "e_gold_at_15": 5000.0 + sign * edge / 2 + rng.normal(0, 150),
                }
            )
            rows.append(
                {
                    "match_id": f"NA1_{index}",
                    "team_id": team,
                    "puuid": f"jg{team}_{index % 5}",
                    "position": "JUNGLE",
                    "champion_name": "Hecarim" if team == 100 else "Belveth",
                    "e_gold_at_15": 5000.0 + rng.normal(0, 150),
                }
            )
    return pd.DataFrame(rows)


def test_lane_pairs_matches_opposing_teams_only():
    pairs = lane_pairs(build_participations(10))
    assert len(pairs) == 40
    assert (pairs["team_id"] != pairs["team_id_enemy"]).all()
    assert (pairs["position"] == pairs["position"]).all()


def test_matchup_edge_is_leave_one_out():
    frame = build_participations()
    edges = matchup_edges(frame)
    single = edges[edges["matchup_games"] == 1]
    assert (single["edge"] == 0.0).all()


def test_edge_recovers_a_planted_counter():
    edges = matchup_edges(build_participations())
    tops = edges[edges["position"] == "TOP"]
    darius = tops[tops["champion_name"] == "Darius"]["total_edge"].mean()
    nasus = tops[tops["champion_name"] == "Nasus"]["total_edge"].mean()
    assert darius > 0 > nasus
    assert darius - nasus > 300


def test_leave_one_out_edge_predicts_the_held_out_game():
    state = counter_state(build_participations())
    assert state["total_edge"].corr(state["lane_gold"]) > 0.5


def test_counter_state_labels_are_terciles_within_role():
    state = counter_state(build_participations())
    for _, group in state.groupby("position"):
        share = (group["lane_state"] == "countered").mean()
        assert 0.2 < share < 0.45


def test_with_ally_state_attaches_the_jungler_and_drops_junglers():
    joined = with_ally_state(counter_state(build_participations()))
    assert "jungle_state" in joined.columns
    assert (joined["position"] != "JUNGLE").all()
    assert joined["jungle_champion"].isin(["Hecarim", "Belveth"]).all()


def test_prior_table_reports_lift_against_the_base_rate():
    frame = pd.DataFrame(
        {
            "context": ["a"] * 100 + ["b"] * 100,
            "outcome": [1.0] * 80 + [0.0] * 20 + [1.0] * 20 + [0.0] * 80,
        }
    )
    table = prior_table(frame, "outcome", ["context"])
    assert table.loc["a", "rate"] == pytest.approx(0.8)
    assert table.loc["b", "rate"] == pytest.approx(0.2)
    assert table.loc["a", "lift"] == pytest.approx(0.3)
    assert abs(table.loc["a", "z"]) > 2


def test_prior_table_drops_thin_cells():
    frame = pd.DataFrame({"context": ["a"] * 100 + ["b"] * 3, "outcome": [1.0] * 103})
    assert list(prior_table(frame, "outcome", ["context"]).index) == ["a"]


def test_player_lift_is_measured_against_the_context_prior():
    rows = []
    for index in range(200):
        context = "hard" if index % 2 else "easy"
        base = 0.2 if context == "hard" else 0.8
        rows.append({"puuid": "always_hard", "context": "hard", "outcome": 1.0})
        rows.append({"puuid": "average", "context": context, "outcome": float(index % 5 < base * 5)})
    lifts = player_lift(pd.DataFrame(rows), "outcome", ["context"])
    assert lifts.loc["always_hard", "lift"] > lifts.loc["average", "lift"]
    assert lifts.loc["always_hard", "prior_rate"] < lifts.loc["always_hard", "rate"]


def test_player_lift_drops_players_below_the_chance_floor():
    frame = pd.DataFrame(
        {"puuid": ["a"] * 10 + ["b"] * 2, "context": ["x"] * 12, "outcome": [1.0] * 12}
    )
    assert list(player_lift(frame, "outcome", ["context"], min_chances=4).index) == ["a"]


def test_separable_reports_no_spread_when_players_are_identical():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "puuid": np.repeat([f"p{i}" for i in range(40)], 30),
            "context": "x",
            "outcome": rng.binomial(1, 0.4, 1200).astype(float),
        }
    )
    assert separable(frame, "outcome", ["context"])["separable"] is False


def test_separable_detects_a_planted_player_difference():
    rng = np.random.default_rng(2)
    rates = np.linspace(0.05, 0.95, 40)
    frame = pd.DataFrame(
        {
            "puuid": np.repeat([f"p{i}" for i in range(40)], 30),
            "context": "x",
            "outcome": np.concatenate([rng.binomial(1, r, 30) for r in rates]).astype(float),
        }
    )
    assert separable(frame, "outcome", ["context"])["separable"] is True
