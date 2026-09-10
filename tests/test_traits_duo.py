import numpy as np
import pandas as pd
import pytest

from synergy.config import Settings
from synergy.features.duo import consistent_duos, fit_duo_effect, pair_key, teammate_pairs
from synergy.features.traits import TRAIT_NAMES, fit_traits, load_traits


def build_corpus(matches: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = [f"p{index}" for index in range(70)]
    traits = {name: rng.normal(0, 1) for name in players}
    rows = []
    for match in range(matches):
        picked = list(rng.choice(players, 10, replace=False))
        for slot, puuid in enumerate(picked):
            team = 100 if slot < 5 else 200
            rows.append(
                {
                    "match_id": f"M{match}",
                    "team_id": team,
                    "puuid": puuid,
                    "position": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][slot % 5],
                    "champion_name": f"C{slot % 7}",
                    "win": int(team == 100),
                    "lp_value": 2800 + 200 * traits[puuid],
                    "e_wards_pm": traits[puuid] + rng.normal(0, 0.3),
                    "e_damage_done_pm": -traits[puuid] + rng.normal(0, 0.3),
                    "e_cs_at_15": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


COLUMNS = ["e_wards_pm", "e_damage_done_pm", "e_cs_at_15"]


@pytest.fixture
def corpus():
    return build_corpus()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


def test_teammate_pairs_are_within_a_team_only(corpus):
    pairs = teammate_pairs(corpus)
    teams = corpus.set_index(["match_id", "puuid"]).team_id
    for _, row in pairs.head(200).iterrows():
        assert teams[(row.match_id, row.puuid_a)] == teams[(row.match_id, row.puuid_b)]


def test_a_team_of_five_yields_ten_pairs(corpus):
    pairs = teammate_pairs(corpus)
    assert len(pairs) == corpus.match_id.nunique() * 20


def test_pair_key_is_order_independent():
    assert pair_key("b", "a") == pair_key("a", "b")


def test_consistent_duos_respect_the_threshold(corpus):
    loose = consistent_duos(corpus, min_together=2)
    tight = consistent_duos(corpus, min_together=20)
    assert tight <= loose


def test_duo_effect_recovers_a_planted_advantage(settings):
    rng = np.random.default_rng(3)
    rows = []
    partners = ("dA", "dB")
    for match in range(600):
        others = [f"q{index}" for index in rng.choice(600, 8, replace=False)]
        blue = list(partners) + others[:3]
        red = others[3:8]
        win_blue = int(rng.random() < 0.75)
        for team, members in ((100, blue), (200, red)):
            for puuid in members:
                rows.append({"match_id": f"M{match}", "team_id": team, "puuid": puuid,
                             "win": int((team == 100) == bool(win_blue)),
                             "lp_value": 2800.0})
    frame = pd.DataFrame(rows)
    out = fit_duo_effect(frame, min_together=5, settings=settings)
    assert out["duo_weight"] > 0
    assert out["duo_sigma"] > 2


def test_duo_effect_is_flat_when_no_duo_helps(settings):
    rng = np.random.default_rng(4)
    rows = []
    partners = ("dA", "dB")
    for match in range(600):
        others = [f"q{index}" for index in rng.choice(600, 8, replace=False)]
        picked = list(partners) + others[:3] + others[3:8]
        win_blue = int(rng.random() < 0.5)
        for slot, puuid in enumerate(picked):
            team = 100 if slot < 5 else 200
            rows.append({"match_id": f"M{match}", "team_id": team, "puuid": puuid,
                         "win": int((team == 100) == bool(win_blue)), "lp_value": 2800.0})
    out = fit_duo_effect(pd.DataFrame(rows), min_together=5, settings=settings)
    assert abs(out["duo_sigma"]) < 3


def test_duo_effect_returns_nothing_without_duos(settings):
    frame = build_corpus(matches=5)
    assert fit_duo_effect(frame, min_together=99, settings=settings) == {}


def test_traits_are_written_and_reloadable(corpus, settings):
    out = fit_traits(corpus, COLUMNS, settings)
    assert set(out["axes"]) <= set(TRAIT_NAMES)
    assert load_traits(settings)["axes"] == out["axes"]


def test_traits_recover_the_planted_player_dimension(corpus, settings):
    out = fit_traits(corpus, COLUMNS, settings)
    best = max(out["reliability"], key=lambda name: out["reliability"][name])
    loadings = out["axes"][best]
    assert abs(loadings.get("e_wards_pm", 0.0)) > abs(loadings.get("e_cs_at_15", 0.0))


def test_the_noise_feature_never_dominates_the_top_axis(corpus, settings):
    out = fit_traits(corpus, COLUMNS, settings)
    best = max(out["reliability"], key=lambda name: out["reliability"][name])
    assert abs(out["axes"][best].get("e_cs_at_15", 0.0)) < 1.0


def test_traits_decline_on_a_corpus_too_small_to_fit(settings):
    assert fit_traits(build_corpus(matches=10), COLUMNS, settings) == {}
