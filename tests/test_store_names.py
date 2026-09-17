from synergy.config import Settings
from synergy.ingest.store import Store


def test_riot_ids_list_each_current_name_and_every_older_one(tmp_path, fresh_database_url):
    # given
    settings = Settings(data_dir=tmp_path, database_url=fresh_database_url)
    settings.ensure_dirs()
    with Store(settings) as store:
        store.upsert_player("p1", game_name="Now", tag_line="NA1")
        store.upsert_player("p2")
        with store._tx() as cursor:
            cursor.execute(
                "INSERT INTO player_names (puuid, game_name, tag_line) VALUES (%s, %s, %s)", ("p1", "Then", "NA1")
            )

        # when
        rows = store.riot_ids()

    # then
    assert sorted((row["puuid"], row["game_name"], row["tag_line"], row["current"]) for row in rows) == [
        ("p1", "Now", "NA1", True),
        ("p1", "Then", "NA1", False),
    ]
