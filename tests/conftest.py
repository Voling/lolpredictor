import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import psycopg

from synergy.config import Settings
from synergy.features.build import build_tables
from synergy.ingest.synthetic import generate
from synergy.ingest.window import build_windows


def _make_database(name: str):
    base = os.environ.get("DATABASE_URL", "postgresql://synergy:synergy@localhost:5432/synergy")
    admin = psycopg.connect(base, autocommit=True)
    with admin.cursor() as cursor:
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
        cursor.execute(f'CREATE DATABASE "{name}"')
    admin.close()
    yield base.rsplit("/", 1)[0] + "/" + name
    admin = psycopg.connect(base, autocommit=True)
    with admin.cursor() as cursor:
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture(scope="session")
def database_url():
    yield from _make_database(f"synergy_test_{os.getpid()}")


@pytest.fixture(scope="session")
def crawl_database_url():
    yield from _make_database(f"synergy_crawl_{os.getpid()}")


@pytest.fixture
def fresh_database_url(request):
    yield from _make_database(f"synergy_{request.node.name[:40]}_{os.getpid()}")


@pytest.fixture(autouse=True)
def hand_axes(tmp_path):
    from synergy.features.player import _refresh_axes
    from synergy.ml.dataset import refresh_columns

    _refresh_axes(Settings(data_dir=tmp_path))
    refresh_columns()


@pytest.fixture(scope="session")
def corpus_settings(tmp_path_factory, database_url) -> Settings:
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("corpus"),
        database_url=database_url,
        min_average_lp=0,
    )
    settings.ensure_dirs()
    generate(matches=60, players=40, duos=6, seed=5, settings=settings)
    build_windows(settings)
    return settings


@pytest.fixture(scope="session")
def tables(corpus_settings):
    return build_tables(corpus_settings)


@pytest.fixture
def serving_settings(tmp_path) -> Settings:
    import numpy as np
    import pandas as pd

    from synergy.ml.serving import TEAM_PAIRS

    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    np.savez(
        settings.model_dir / "interaction_matrix.npz",
        matrix=np.array([[1.0, 0.0], [0.0, -1.0]]),
        columns=np.array(["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]),
        centre=np.zeros(2),
        spread=np.ones(2),
    )
    pd.DataFrame(
        {
            "puuid": ["a", "a", "b", "c", "d", "e"],
            "position": ["TOP", "JUNGLE", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
            "tend_dive_tmb_own": [1.0, 0.0, 1.0, 0.0, 1.0, 0.5],
            "rsp_kill_ours_near_converged": [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
            "seats": [5, 1, 7, 2, 3, 4],
            "evidence": [0.8, 0.1, 0.9, 0.4, 0.6, 0.7],
        }
    ).to_parquet(settings.processed_dir / "player_styles.parquet", index=False)
    grid = np.linspace(-1.0, 1.0, 1001) / (TEAM_PAIRS * 2)
    np.savez(
        settings.model_dir / "interaction_scores.npz",
        quantiles=grid,
        combos=np.array(["JUNGLE+TOP", "MIDDLE+TOP"]),
        combo_quantiles=np.stack([grid, grid * 4.0]),
        team_quantiles=grid * 10.0,
        gold_sd=5000.0,
    )
    return settings
