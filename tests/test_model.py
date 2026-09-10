import random

import numpy as np
import pandas as pd
import pytest

from synergy.features.player import STYLE_COLUMNS, build_profiles
from synergy.ingest.synthetic import make_players, true_synergy
from synergy.ml.dataset import (
    PHI_COLUMNS,
    STYLE_NAMES,
    build_pair_dataset,
    build_team_dataset,
    leave_one_out,
    player_context,
)
from synergy.ml.model import SynergyModel
from synergy.ml.score import SynergyService, UnknownPlayer


def test_leave_one_out_excludes_the_row_itself():
    frame = pd.DataFrame({"key": ["a", "a", "a", "b"], "value": [1.0, 2.0, 3.0, 9.0]})
    out = leave_one_out(frame, "key", ["value"])["value"].tolist()
    assert out[0] == pytest.approx(2.5)
    assert out[1] == pytest.approx(2.0)
    assert np.isnan(out[3])


def test_profiles_summarise_every_player(tables, corpus_settings):
    profiles = build_profiles(tables["participations"], min_games=2, settings=corpus_settings)
    assert len(profiles) > 20
    assert set(STYLE_COLUMNS).issubset(profiles.columns)
    assert profiles["games"].min() >= 2
    assert profiles[[f"{column}_pct" for column in STYLE_COLUMNS]].max().max() <= 100
    assert not any(
        column in profiles.columns
        for column in ("streakiness", "matches_per_session", "blue_winrate", "surrender_rate")
    )


def test_pair_dataset_has_one_row_per_team_pair(tables):
    features, controls, _ = build_pair_dataset(tables["participations"], tables["pairs"])
    matches = features["match_id"].nunique()
    assert len(features) == matches * 20
    assert set(PHI_COLUMNS).issubset(features.columns)
    assert controls.groupby("match_id").size().max() == 2


def test_pair_features_are_symmetric(tables):
    features, _, _ = build_pair_dataset(tables["participations"], tables["pairs"])
    row = features.iloc[0]
    assert row["puuid_a"] < row["puuid_b"]
    assert (features[f"diff_{STYLE_NAMES[0]}"] >= 0).all()


def test_team_dataset_is_a_signed_difference(tables):
    features, controls, _ = build_pair_dataset(tables["participations"], tables["pairs"])
    X, y, observed = build_team_dataset(features, controls)
    assert len(X) == features["match_id"].nunique()
    assert set(y) <= {0, 1}
    assert X.abs().to_numpy().sum() > 0
    assert len(observed) == len(X)
    assert observed.max() <= 10


def test_scores_are_percentiles_of_the_reference_distribution():
    model = SynergyModel()
    model.weights = pd.Series(1.0, index=PHI_COLUMNS)
    model.quantiles = np.linspace(-1.0, 1.0, 1001)
    assert model.score(-2.0)[0] == 0.0
    assert model.score(2.0)[0] == 100.0
    assert model.score(0.0)[0] == pytest.approx(50.0, abs=0.2)


def test_model_recovers_planted_synergy(tmp_path, database_url):
    from synergy.config import Settings
    from synergy.features.build import build_tables
    from synergy.ingest.synthetic import generate
    from synergy.ingest.window import build_windows
    from synergy.ml.train import train

    settings = Settings(data_dir=tmp_path, database_url=database_url)
    settings.ensure_dirs()
    generate(matches=900, players=90, duos=12, seed=13, settings=settings)
    build_windows(settings)
    build_tables(settings)
    train(settings, min_games=3)

    service = SynergyService(settings).load()
    latent = {player["puuid"]: player for player in make_players(90, random.Random(13), settings.seed_riot_id)}
    rng = random.Random(3)
    keys = [key for key in service.profiles.index if key in latent]
    sampled = [rng.sample(keys, 2) for _ in range(400)]
    predicted, actual = [], []
    for left, right in sampled:
        phi = service.build_phi(service.profiles.loc[left], service.profiles.loc[[right]])
        predicted.append(float(service.model.synergy(phi)[0]))
        actual.append(true_synergy(latent[left], latent[right]))
    correlation = np.corrcoef(predicted, actual)[0, 1]
    assert correlation > 0.15


def test_service_rejects_unknown_players(corpus_settings):
    service = SynergyService(corpus_settings)
    with pytest.raises(RuntimeError):
        service.resolve("nobody#none")
    service.load()
    if service.ready:
        with pytest.raises(UnknownPlayer):
            service.resolve("nobody#none")


def test_sparse_players_are_shrunk_to_the_population_average(tables):
    participations = tables["participations"]
    context, _ = player_context(participations, shrinkage_k=5.0)
    counts = context.groupby("puuid")["puuid"].transform("count")
    single = context[counts == 1]
    if not single.empty:
        assert (single[[f"loo_style_{name}" for name in STYLE_NAMES]].abs() < 1e-9).all().all()
        assert (single["loo_winrate"] - 0.5).abs().max() < 1e-9
    heavy = context[counts >= 20]
    assert heavy[[f"loo_style_{name}" for name in STYLE_NAMES]].abs().to_numpy().max() > 0.05


def test_shrinkage_scales_with_the_number_of_games():
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "puuid": ["few", "few", "many", "many"],
            "team_id": [100, 100, 100, 100],
            "position": ["MIDDLE"] * 4,
            "champion_name": ["Ahri"] * 4,
            "win": [1, 0, 1, 0],
            "duration_min": [30.0] * 4,
            "e_damage_done_pm": [0.4, 0.4, 0.4, 0.4],
        }
    )
    light, _ = player_context(frame, shrinkage_k=1.0)
    heavy, _ = player_context(frame, shrinkage_k=50.0)
    column = "loo_style_aggression"
    assert abs(heavy[column]).max() <= abs(light[column]).max()


def test_profiles_report_style_confidence(tables, corpus_settings):
    profiles = build_profiles(tables["participations"], min_games=2, settings=corpus_settings)
    assert "style_confidence" in profiles.columns
    assert profiles["style_confidence"].between(0, 1).all()
    ordered = profiles.sort_values("games")
    assert ordered["style_confidence"].is_monotonic_increasing
