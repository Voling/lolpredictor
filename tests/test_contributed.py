import copy
import gzip
import io
import json

import pytest

from synergy.config import Settings
from synergy.ingest.contributed import LocalStore, MissingKey, S3Store, basic_reason, contribute, rank_reason
from synergy.ingest.lock import CorpusBusy, corpus_lock
from synergy.ingest.store import Store, batch_filter
from synergy.ingest.synthetic import generate


def _game(queue: int = 420, version: str = "16.3.1", result: str = "GameComplete", platform: str = "NA1") -> dict:
    players = [{"puuid": f"p{index}", "gameEndedInEarlySurrender": False} for index in range(10)]
    return {"info": {"platformId": platform, "queueId": queue, "gameVersion": version, "endOfGameResult": result, "participants": players}}


def _write(folder, kind: str, match_id: str, payload: dict) -> None:
    (folder / kind).mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / kind / f"{match_id}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_the_batch_filter_keeps_everything_the_crawl_alone_or_named_batches():
    # given
    selections = ["all", "crawl", "crawl,batch-20260927-170000"]

    # when
    found = [batch_filter(selection) for selection in selections]

    # then
    assert found[0] == ("TRUE", [])
    assert found[1] == ("(m.source = 'crawl' OR m.batch = ANY(%s))", [[]])
    assert found[2] == ("(m.source = 'crawl' OR m.batch = ANY(%s))", [["batch-20260927-170000"]])
    with pytest.raises(ValueError):
        batch_filter("crawl; DROP TABLE matches")


def test_a_contributed_game_must_match_the_corpus_platform_queue_season_and_finish():
    # given
    settings = Settings(season=16, queue_id=420, platform="na1")
    games = [_game(), _game(platform="EUW1"), _game(queue=440), _game(version="15.24.1"), _game(result="Abort_Unexpected")]

    # when
    reasons = [basic_reason(game, settings) for game in games]

    # then
    assert reasons == [None, "another platform", "another queue", "another season", "unfinished"]


def test_a_contributed_game_needs_enough_ranked_players_averaging_master():
    # given
    settings = Settings(min_ranked_participants=2, min_average_lp=2800)
    players = [f"p{index}" for index in range(10)]

    # when
    lone = rank_reason(players, {"p0": 3400}, settings)
    below = rank_reason(players, {"p0": 3000, "p1": 2500}, settings)
    master = rank_reason(players, {"p0": 3000, "p1": 2700}, settings)

    # then
    assert lone == "too few ranked players"
    assert below == "average below 2800 LP"
    assert master is None


def test_a_folder_store_lists_only_games_with_both_match_and_timeline(tmp_path):
    # given
    _write(tmp_path, "matches", "NA1_1", {"id": 1})
    _write(tmp_path, "timelines", "NA1_1", {"frames": []})
    _write(tmp_path, "matches", "NA1_2", {"id": 2})

    # when
    source = LocalStore(tmp_path)

    # then
    assert source.ids() == ["NA1_1"]
    assert source.read("matches", "NA1_1") == {"id": 1}


class _Pages:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, Bucket, Prefix):
        return [{"Contents": [item for item in page if item["Key"].startswith(Prefix)]} for page in self.pages]


class _Bucket:
    def __init__(self, objects: dict[str, bytes], stuck: tuple = ()):
        self.objects, self.stuck = objects, stuck
        keys = [{"Key": key} for key in objects]
        self.pages = [keys[:2], keys[2:]]

    def get_paginator(self, name):
        return _Pages(self.pages)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_objects(self, Bucket, Delete):
        errors = []
        for item in Delete["Objects"]:
            if item["Key"] in self.stuck:
                errors.append({"Key": item["Key"], "Code": "AccessDenied"})
            else:
                self.objects.pop(item["Key"], None)
        return {"Errors": errors} if errors else {}


def test_an_s3_store_pages_through_keys_and_reads_gzipped_games():
    # given
    body = gzip.compress(json.dumps({"id": 7}).encode("utf-8"))
    client = _Bucket({
        "contributed/matches/NA1_7.json.gz": body,
        "contributed/timelines/NA1_7.json.gz": body,
        "contributed/matches/NA1_8.json.gz": body,
    })

    # when
    source = S3Store("games", "contributed/", client)

    # then
    assert source.ids() == ["NA1_7"]
    assert source.read("timelines", "NA1_7") == {"id": 7}
    source.delete(["NA1_7"])
    assert list(client.objects) == ["contributed/matches/NA1_8.json.gz"]


@pytest.fixture
def small_corpus(tmp_path, fresh_database_url):
    settings = Settings(data_dir=tmp_path / "corpus", database_url=fresh_database_url, min_average_lp=2800)
    settings.ensure_dirs()
    generate(matches=6, players=20, duos=2, seed=3, settings=settings)
    store = Store(settings)
    original = store.match_ids()[0]
    match, timeline = store.load_match(original), store.load_timeline(original)
    contributed = tmp_path / "contributed"
    for match_id, queue in (("NA1_9000000001", settings.queue_id), ("NA1_9000000002", 440)):
        clone = copy.deepcopy(match)
        clone["metadata"]["matchId"] = match_id
        clone["info"]["queueId"] = queue
        _write(contributed, "matches", match_id, clone)
        _write(contributed, "timelines", match_id, timeline)
    for participant in match["info"]["participants"]:
        store.upsert_player(participant["puuid"], lp_value=3000)
    store.close()
    return settings, LocalStore(contributed)


def test_a_confirmed_import_adds_only_qualified_games_as_one_batch_the_pipeline_can_leave_out(small_corpus):
    # given
    settings, source = small_corpus
    asked = []

    # when
    found = contribute(settings, lambda summary, batch: asked.append(summary) or True, source=source, lookup=lambda puuids: {})

    # then
    assert asked == [{"waiting": 2, "in corpus": 0, "qualified": 1, "another queue": 1}]
    assert found["imported"] == 1 and found["batch"].startswith("batch-")
    assert found["removed"] == 2 and source.ids() == []
    everything = Store(settings)
    crawl_only = Store(Settings(data_dir=settings.data_dir, database_url=settings.database_url, corpus_batches="crawl"))
    try:
        assert "NA1_9000000001" in everything.match_ids() and "NA1_9000000002" not in everything.match_ids()
        assert "NA1_9000000001" not in crawl_only.corpus_ids()
        assert everything.batch_counts() == {"crawl": 6, found["batch"]: 1}
        assert crawl_only.batch_counts() == {"crawl": 6}
        with everything.conn.cursor() as cursor:
            cursor.execute("SELECT count(*) AS frames FROM frames WHERE match_id = 'NA1_9000000001'")
            assert cursor.fetchone()["frames"] > 0
            cursor.execute("SELECT matches, finished_at IS NOT NULL AS done FROM batches WHERE batch = %s", (found["batch"],))
            assert cursor.fetchone() == {"matches": 1, "done": True}
    finally:
        everything.close()
        crawl_only.close()


def test_a_declined_import_leaves_the_corpus_untouched(small_corpus):
    # given
    settings, source = small_corpus

    # when
    found = contribute(settings, lambda summary, batch: False, source=source, lookup=lambda puuids: {})

    # then
    store = Store(settings)
    try:
        assert found["imported"] == 0 and found["removed"] == 0 and "batch" not in found
        assert len(source.ids()) == 2
        assert store.batch_counts() == {"crawl": 6}
    finally:
        store.close()


def test_an_import_waits_its_turn_while_a_pipeline_run_holds_the_corpus(small_corpus):
    # given
    settings, source = small_corpus

    # when
    with corpus_lock(settings, "a pipeline run"):
        with pytest.raises(CorpusBusy) as busy:
            contribute(settings, lambda summary, batch: True, source=source, lookup=lambda puuids: {})

    # then
    assert "a pipeline run" in str(busy.value)


def test_names_that_are_not_riot_match_ids_never_reach_a_path(tmp_path):
    # given
    for name in ("NA1_5123456789", r"..\evil", "NA1_12;rm", "na1_123"):
        _write(tmp_path, "matches", name, {"id": name})
        _write(tmp_path, "timelines", name, {"frames": []})

    # when
    listed = LocalStore(tmp_path).ids()

    # then
    assert listed == ["NA1_5123456789"]


def test_a_game_whose_body_names_another_match_is_refused(small_corpus):
    # given
    settings, source = small_corpus
    forged = source.read("matches", "NA1_9000000001")
    forged["metadata"]["matchId"] = "NA1_1234567890"
    _write(source.root, "matches", "NA1_9000000001", forged)

    # when
    found = contribute(settings, lambda summary, batch: False, source=source, lookup=lambda puuids: {})

    # then
    assert found["mismatched id"] == 1 and found["qualified"] == 0


def test_the_import_stops_before_judging_when_it_cannot_rank_unknown_players(small_corpus):
    # given
    settings, source = small_corpus
    keyless = Settings(data_dir=settings.data_dir, database_url=settings.database_url, min_average_lp=2800, riot_api_key="")
    store = Store(keyless)
    with store.conn.cursor() as cursor:
        cursor.execute("UPDATE players SET lp_value = NULL")
    store.conn.commit()
    store.close()

    # when
    with pytest.raises(MissingKey):
        contribute(keyless, lambda summary, batch: True, source=source)

    # then
    assert len(source.ids()) == 2


def test_an_s3_delete_that_keeps_objects_fails_loudly():
    # given
    body = gzip.compress(json.dumps({"id": 7}).encode("utf-8"))
    keys = ["contributed/matches/NA1_7.json.gz", "contributed/timelines/NA1_7.json.gz"]
    source = S3Store("games", "contributed/", _Bucket(dict.fromkeys(keys, body), stuck=(keys[1],)))

    # when
    with pytest.raises(RuntimeError) as kept:
        source.delete(["NA1_7"])

    # then
    assert "kept 1 objects" in str(kept.value)


def test_a_game_without_a_start_time_is_refused_and_cleared(small_corpus):
    # given
    settings, source = small_corpus
    broken = source.read("matches", "NA1_9000000001")
    del broken["info"]["gameCreation"]
    _write(source.root, "matches", "NA1_9000000001", broken)

    # when
    found = contribute(settings, lambda summary, batch: True, source=source, lookup=lambda puuids: {})

    # then
    assert found["no start time"] == 1 and found["imported"] == 0 and source.ids() == []


def test_an_import_that_crashes_leaves_the_store_and_the_next_run_finishes_the_game(small_corpus, monkeypatch):
    # given
    settings, source = small_corpus
    real = LocalStore.read

    def flaky(self, kind, match_id):
        if kind == "timelines":
            raise OSError("disk hiccup")
        return real(self, kind, match_id)

    monkeypatch.setattr(LocalStore, "read", flaky)
    with pytest.raises(OSError):
        contribute(settings, lambda summary, batch: True, source=source, lookup=lambda puuids: {})
    monkeypatch.setattr(LocalStore, "read", real)

    # when
    found = contribute(settings, lambda summary, batch: True, source=source, lookup=lambda puuids: {})

    # then
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute("SELECT source, batch FROM matches WHERE match_id = 'NA1_9000000001'")
            row = cursor.fetchone()
            cursor.execute("SELECT count(*) AS frames FROM frames WHERE match_id = 'NA1_9000000001'")
            frames = cursor.fetchone()["frames"]
    finally:
        store.close()
    assert found["imported"] == 1 and found["in corpus"] == 0 and source.ids() == []
    assert row == {"source": "contributed", "batch": found["batch"]} and frames > 0
