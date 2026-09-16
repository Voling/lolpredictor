import psycopg
import pytest

from synergy.config import Settings
from synergy.ingest.store import Store


@pytest.fixture
def store(tmp_path, fresh_database_url):
    settings = Settings(data_dir=tmp_path, database_url=fresh_database_url)
    settings.ensure_dirs()
    with Store(settings) as opened:
        yield opened


def test_a_dropped_connection_is_replaced_so_the_next_write_succeeds(store):
    # given
    store.upsert_player("before-drop", tier="MASTER")
    store.conn.close()

    # when
    with pytest.raises(psycopg.OperationalError):
        store.upsert_player("during-drop", tier="MASTER")

    # then
    store.upsert_player("after-drop", tier="MASTER")
    assert store.get_player("after-drop")["tier"] == "MASTER"
    assert store.get_player("before-drop")["tier"] == "MASTER"
