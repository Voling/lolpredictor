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


@pytest.fixture(scope="session")
def database_url() -> str:
    base = os.environ.get("DATABASE_URL", "postgresql://synergy:synergy@localhost:5432/synergy")
    name = f"synergy_test_{os.getpid()}"
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
