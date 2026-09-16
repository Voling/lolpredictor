import pandas as pd
import pytest

from synergy.config import Settings
from synergy.ingest.breadth import enqueue, targets
from synergy.ingest.store import Store


@pytest.fixture
def seeded(tmp_path, fresh_database_url):
    settings = Settings(data_dir=tmp_path, database_url=fresh_database_url, riot_api_key="test-key")
    settings.ensure_dirs()
    rows = []
    for puuid, position, games in (
        ("apex-thin", "TOP", 12),
        ("apex-deep", "JUNGLE", 40),
        ("apex-shallow", "MIDDLE", 4),
        ("diamond-deep", "TOP", 30),
    ):
        rows += [{"match_id": f"{puuid}-{n}", "puuid": puuid, "position": position} for n in range(games)]
    pd.DataFrame(rows).to_parquet(settings.processed_dir / "participations.parquet", index=False)
    with Store(settings) as store:
        for puuid, tier in (
            ("apex-thin", "MASTER"),
            ("apex-deep", "CHALLENGER"),
            ("apex-shallow", "GRANDMASTER"),
            ("diamond-deep", "DIAMOND"),
        ):
            store.upsert_player(puuid, tier=tier)
    return settings


def test_only_master_and_above_with_real_depth_qualify(seeded):
    # given
    minimum = 10

    # when
    wanted = targets(seeded, min_games=minimum)

    # then
    assert wanted.puuid.tolist() == ["apex-thin", "apex-deep"]
    assert wanted.games.tolist() == [12, 40]


def test_queueing_puts_the_thinnest_player_at_the_front(seeded):
    # given
    with Store(seeded) as store:
        store.push_frontier("apex-deep", depth=0, priority=1.0)

    # when
    report = enqueue(seeded, min_games=10)
    with Store(seeded) as store:
        popped = [entry["puuid"] for entry in store.pop_frontier(2)]

    # then
    assert report["queued"] == 2 and report["thinnest"] == 12 and report["deepest"] == 40
    assert popped[0] == "apex-thin"
