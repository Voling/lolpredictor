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

    from synergy.features.positions import POSITIONS

    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    columns = np.array(["tend_dive_tmb_own", "rsp_kill_ours_near_converged"])
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
    combos = [f"{POSITIONS[a]}+{POSITIONS[b]}" for a in range(5) for b in range(a + 1, 5)]
    grid = np.linspace(-1.0, 1.0, 1001)
    reference = np.tile(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32), (5, 1, 1))
    np.savez(
        settings.model_dir / "pairnet.npz",
        first_weight=np.array([[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        first_bias=np.array([[10.0, 10.0]], dtype=np.float32),
        second_weight=np.array([np.eye(2)], dtype=np.float32),
        second_bias=np.zeros((1, 2), dtype=np.float32),
        third_weight=np.array([[1.0, -1.0]], dtype=np.float32),
        third_bias=np.zeros(1, dtype=np.float32),
        scale=np.ones(1),
        columns=columns,
        centre=np.zeros(2),
        spread=np.ones(2),
        reference=reference,
        positions=np.array(list(POSITIONS)),
        combos=np.array(combos),
        grand=np.zeros(len(combos)),
        grand_seeds=np.zeros((len(combos), 1)),
        families=np.array(["rsp", "tend"]),
        grand_without=np.zeros((len(combos), 2)),
        fit_quantiles=np.tile(grid, (len(combos), 5, 1)),
        evidence_bands=np.array([0.0, 0.02, 0.08, 0.2, 0.35]),
        team_quantiles=grid * 10.0,
        ridge_weights=np.zeros(2),
        ridge_middle=0.0,
        units=np.array("gold at 15"),
    )
    np.savez(
        settings.model_dir / "seat_weights.npz",
        weights=np.array([[100.0, 0.0], [50.0, 0.0], [0.0, 40.0], [30.0, 0.0], [0.0, 20.0]]),
        centres=np.zeros((5, 2)),
        quantiles=np.tile(np.linspace(-200.0, 200.0, 1001), (5, 1)),
        columns=columns,
        positions=np.array(list(POSITIONS)),
    )
    return settings
