import asyncio
import json

import httpx
import pytest

from synergy.config import Settings
from synergy.ingest.seeder import Seeder
from synergy.ingest.store import Store
from synergy.riot.client import NotFound, RiotClient

TIERS = {
    "puuid-seed": "DIAMOND",
    "puuid-1": "MASTER",
    "puuid-2": "DIAMOND",
    "puuid-3": "GOLD",
}


def make_match(match_id: str, puuids: list[str], duration: int = 1800, queue: int = 420) -> dict:
    participants = [
        {
            "participantId": index + 1,
            "puuid": puuid,
            "teamId": 100 if index < 5 else 200,
            "teamPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][index % 5],
            "championId": 1,
            "championName": "Annie",
            "win": index < 5,
            "kills": 1,
            "deaths": 1,
            "assists": 1,
        }
        for index, puuid in enumerate(puuids)
    ]
    return {
        "metadata": {"matchId": match_id, "participants": puuids},
        "info": {
            "gameCreation": 1735689600000,
            "gameDuration": duration,
            "gameVersion": "15.1.1.1",
            "queueId": queue,
            "platformId": "NA1",
            "participants": participants,
        },
    }


class FakeRiot:
    def __init__(self):
        self.roster = ["puuid-seed", "puuid-1", "puuid-2", "puuid-3"] + [f"puuid-x{i}" for i in range(6)]
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if "by-riot-id" in path:
            return httpx.Response(200, json={"puuid": "puuid-seed", "gameName": "seed", "tagLine": "na1"})
        if "by-puuid" in path and "accounts" in path:
            puuid = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"puuid": puuid, "gameName": puuid, "tagLine": "na1"})
        if "league/v4/entries" in path:
            puuid = path.rsplit("/", 1)[-1]
            tier = TIERS.get(puuid, "DIAMOND")
            return httpx.Response(
                200,
                json=[
                    {
                        "queueType": "RANKED_SOLO_5x5",
                        "tier": tier,
                        "rank": "II",
                        "leaguePoints": 40,
                        "wins": 60,
                        "losses": 50,
                    }
                ],
            )
        if path.endswith("/ids"):
            puuid = path.split("/by-puuid/")[1].split("/")[0]
            return httpx.Response(200, json=[f"NA1_{puuid}_0"])
        if path.endswith("/timeline"):
            return httpx.Response(200, json={"metadata": {"matchId": "x"}, "info": {"frames": []}})
        if "/matches/" in path:
            match_id = path.rsplit("/", 1)[-1]
            owner = match_id.split("_")[1]
            others = [p for p in self.roster if p != owner][:9]
            return httpx.Response(200, json=make_match(match_id, [owner, *others]))
        return httpx.Response(404, json={"status": {"status_code": 404}})


def run_crawl(settings: Settings, fake: FakeRiot):
    async def main():
        with Store(settings) as store:
            async with RiotClient(settings, transport=httpx.MockTransport(fake.handler)) as client:
                return await Seeder(store, client, settings).run(), store.counts()

    return asyncio.run(main())


@pytest.fixture
def crawl_settings(tmp_path, crawl_database_url):
    settings = Settings(
        data_dir=tmp_path,
        database_url=crawl_database_url,
        riot_api_key="test-key",
        max_matches=5,
        max_players=4,
        matches_per_player=2,
        max_depth=1,
        rate_short_requests=1000,
        rate_long_requests=1000,
    )
    settings.ensure_dirs()
    with Store(settings) as store:
        with store._tx() as cursor:
            cursor.execute(
                "TRUNCATE matches, players, participations, frames, events, frontier"
                " RESTART IDENTITY CASCADE"
            )
    return settings


def test_crawl_stores_matches_and_discovers_participants(crawl_settings):
    fake = FakeRiot()
    report, counts = run_crawl(crawl_settings, fake)
    assert report.matches_added >= 1
    assert report.timelines_added == report.matches_added
    assert counts["participations"] == counts["matches"] * 10
    assert counts["players"] >= 10


def test_crawl_skips_players_below_the_tier_floor(crawl_settings):
    fake = FakeRiot()
    run_crawl(crawl_settings, fake)
    with Store(crawl_settings) as store:
        gold = store.get_player("puuid-3")
    assert gold is not None
    assert gold["in_scope"] == 0


def test_crawl_ignores_remakes_and_other_queues(crawl_settings):
    class OddQueue(FakeRiot):
        def handler(self, request):
            if "/matches/" in request.url.path and not request.url.path.endswith("/timeline"):
                match_id = request.url.path.rsplit("/", 1)[-1]
                return httpx.Response(200, json=make_match(match_id, self.roster[:10], duration=300))
            return super().handler(request)

    report, counts = run_crawl(crawl_settings, OddQueue())
    assert report.matches_added == 0
    assert counts["matches"] == 0


def test_reseeding_rechecks_the_seed_without_refetching_matches(crawl_settings):
    crawl_settings.max_players = 50
    crawl_settings.max_matches = 50
    run_crawl(crawl_settings, FakeRiot())
    with Store(crawl_settings) as store:
        first_ids = set(store.match_ids())
    report, counts = run_crawl(crawl_settings, FakeRiot())
    with Store(crawl_settings) as store:
        assert set(store.match_ids()) == first_ids
    assert report.matches_added == 0
    assert report.players_crawled == 1


def test_client_raises_not_found_for_unknown_paths(crawl_settings):
    async def main():
        async with RiotClient(crawl_settings, transport=httpx.MockTransport(FakeRiot().handler)) as client:
            with pytest.raises(NotFound):
                await client._get("https://americas.api.riotgames.com/nope")

    asyncio.run(main())


def test_client_retries_after_a_rate_limit(crawl_settings):
    state = {"hits": 0}

    def handler(request):
        state["hits"] += 1
        if state["hits"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"ok": True})

    async def main():
        async with RiotClient(crawl_settings, transport=httpx.MockTransport(handler)) as client:
            return await client._get("https://americas.api.riotgames.com/anything")

    assert asyncio.run(main()) == {"ok": True}
    assert state["hits"] == 2


def test_store_round_trips_raw_json(crawl_settings):
    match = make_match("NA1_ROUND", [f"puuid-{i}" for i in range(10)])
    with Store(crawl_settings) as store:
        store.save_match(match)
        store.save_timeline(
            "NA1_ROUND",
            {
                "metadata": {"matchId": "NA1_ROUND"},
                "info": {
                    "participants": [{"participantId": i + 1, "puuid": f"puuid-{i}"} for i in range(10)],
                    "frames": [
                        {
                            "timestamp": 0,
                            "participantFrames": {
                                str(i + 1): {"participantId": i + 1, "position": {"x": 1, "y": 2}}
                                for i in range(10)
                            },
                            "events": [],
                        }
                    ],
                },
            },
        )
        assert store.has_match("NA1_ROUND")
        assert store.load_match("NA1_ROUND")["metadata"]["matchId"] == "NA1_ROUND"
        assert store.counts()["timelines"] == 1
        raw = store.match_path("NA1_ROUND")
    assert raw.suffix == ".gz"
    assert json.dumps(match)


def test_reseeding_a_finished_player_crawls_them_again(crawl_settings):
    crawl_settings.max_players = 1
    crawl_settings.matches_per_player = 1
    run_crawl(crawl_settings, FakeRiot())
    with Store(crawl_settings) as store:
        assert store.get_player("puuid-seed")["crawled_at"] is not None
        store.conn.execute("DELETE FROM matches")
        store.conn.execute("DELETE FROM participations")
        store.conn.commit()
        for path in crawl_settings.match_dir.glob("*.json.gz"):
            path.unlink()
    report, counts = run_crawl(crawl_settings, FakeRiot())
    assert report.players_crawled == 1
    assert counts["matches"] >= 1


def test_identities_are_read_from_the_match_payload(crawl_settings):
    match = make_match("NA1_IDENT", [f"puuid-{i}" for i in range(10)])
    for index, participant in enumerate(match["info"]["participants"]):
        participant["riotIdGameName"] = f"name{index}"
        participant["riotIdTagline"] = "NA1"
    with Store(crawl_settings) as store:
        store.save_match(match)
        assert store.get_player("puuid-3")["game_name"] == "name3"
        assert store.get_player("puuid-3")["tag_line"] == "NA1"
        assert store.unnamed_players(1) == []


def test_backfill_names_players_without_touching_the_api(crawl_settings):
    match = make_match("NA1_BACKFILL", [f"puuid-{i}" for i in range(10)])
    for index, participant in enumerate(match["info"]["participants"]):
        participant["riotIdGameName"] = f"late{index}"
        participant["riotIdTagline"] = "EUW"
    with Store(crawl_settings) as store:
        store.save_match(match)
        store.conn.execute("UPDATE players SET game_name=NULL, tag_line=NULL")
        store.conn.commit()
        assert store.get_player("puuid-0")["game_name"] is None
        assert store.backfill_identities() == 10
        assert store.get_player("puuid-0")["game_name"] == "late0"


def test_backfill_does_not_overwrite_an_existing_name(crawl_settings):
    match = make_match("NA1_KEEP", [f"puuid-{i}" for i in range(10)])
    for participant in match["info"]["participants"]:
        participant["riotIdGameName"] = "fromjson"
        participant["riotIdTagline"] = "NA1"
    with Store(crawl_settings) as store:
        store.upsert_player("puuid-0", game_name="authoritative", tag_line="OLD")
        store.save_match(match)
        assert store.get_player("puuid-0")["game_name"] == "authoritative"
        assert store.get_player("puuid-1")["game_name"] == "fromjson"


def test_participants_without_a_riot_id_are_skipped(crawl_settings):
    match = make_match("NA1_BLANK", [f"puuid-{i}" for i in range(10)])
    for participant in match["info"]["participants"]:
        participant["riotIdGameName"] = ""
        participant["riotIdTagline"] = ""
    with Store(crawl_settings) as store:
        store.save_match(match)
        assert store.get_player("puuid-0")["game_name"] is None


def _frontier(store, puuid):
    with store._tx() as cursor:
        cursor.execute("SELECT depth, priority, state FROM frontier WHERE puuid=%s", (puuid,))
        return cursor.fetchone()


@pytest.mark.parametrize(
    "first, second, depth",
    [((3, 1.0), (1, 1.0), 1), ((1, 1.0), (3, 1.0), 1)],
    ids=["a shorter path wins", "a longer path is ignored"],
)
def test_discovery_keeps_the_shortest_path_to_a_player(crawl_settings, first, second, depth):
    with Store(crawl_settings) as store:
        store.push_frontier("p", depth=first[0], priority=first[1])
        store.push_frontier("p", depth=second[0], priority=second[1])
        assert _frontier(store, "p")["depth"] == depth


def test_discovery_boosts_priority_for_a_player_seen_again(crawl_settings):
    with Store(crawl_settings) as store:
        store.push_frontier("p", depth=2, priority=5.0)
        store.push_frontier("p", depth=2, priority=5.0)
        assert _frontier(store, "p")["priority"] == 6.0


def test_seeding_requeues_a_finished_player_without_lowering_priority(crawl_settings):
    with Store(crawl_settings) as store:
        store.push_frontier("p", depth=2, priority=1.0)
        store.mark_frontier("p", "done")
        store.push_frontier("p", depth=0, priority=1744.0, requeue=True)
        row = _frontier(store, "p")
        assert (row["depth"], row["priority"], row["state"]) == (0, 1744.0, "pending")
        store.push_frontier("p", depth=0, priority=17.0, requeue=True)
        assert _frontier(store, "p")["priority"] == 1744.0


def test_frontier_pops_the_shallowest_then_highest_priority_first(crawl_settings):
    with Store(crawl_settings) as store:
        store.push_frontier("deep", depth=2, priority=9999.0)
        store.push_frontier("low", depth=0, priority=17.0)
        store.push_frontier("high", depth=0, priority=1744.0)
        assert [row["puuid"] for row in store.pop_frontier(3)] == ["high", "low", "deep"]


def test_a_named_seed_is_crawled_before_a_crowded_frontier(crawl_settings):
    with Store(crawl_settings) as store:
        for index in range(60):
            store.push_frontier(f"crowd{index:04d}", depth=0, priority=1e9)
    fake = FakeRiot()
    report, _ = run_crawl(crawl_settings, fake)
    with Store(crawl_settings) as store:
        assert _frontier(store, "puuid-seed")["state"] == "done"
    assert report.players_crawled >= 1


def test_discovery_can_be_switched_off(crawl_settings):
    crawl_settings.crawl_discover = False
    fake = FakeRiot()
    report, counts = run_crawl(crawl_settings, fake)
    assert report.matches_added >= 1
    assert report.players_discovered == 0
    with Store(crawl_settings) as store:
        with store.conn.cursor() as cursor:
            cursor.execute("SELECT count(*) AS n FROM frontier WHERE depth > 0")
            assert cursor.fetchone()["n"] == 0


def test_a_renamed_player_is_found_by_either_name(crawl_settings):
    first = make_match("NA1_RENAME_OLD", [f"puuid-{i}" for i in range(10)])
    for participant in first["info"]["participants"]:
        participant["riotIdGameName"] = "oldname"
        participant["riotIdTagline"] = "aaaa"
    second = make_match("NA1_RENAME_NEW", [f"puuid-{i}" for i in range(10)])
    for participant in second["info"]["participants"]:
        participant["riotIdGameName"] = "newname"
        participant["riotIdTagline"] = "bbbb"
    with Store(crawl_settings) as store:
        store.save_match(first)
        store.save_match(second)
        by_old = store.find_player_by_riot_id("oldname", "aaaa")
        by_new = store.find_player_by_riot_id("newname", "bbbb")
        assert by_old is not None and by_new is not None
        assert by_old["puuid"] == by_new["puuid"]
        assert store.find_player_by_riot_id("oldname", "bbbb") is None


def test_the_new_name_is_only_reachable_through_the_alias_table(crawl_settings):
    first = make_match("NA1_GUARD_OLD", [f"puuid-g{i}" for i in range(10)])
    for p in first["info"]["participants"]:
        p["riotIdGameName"], p["riotIdTagline"] = "guardold", "aaaa"
    second = make_match("NA1_GUARD_NEW", [f"puuid-g{i}" for i in range(10)])
    for p in second["info"]["participants"]:
        p["riotIdGameName"], p["riotIdTagline"] = "guardnew", "bbbb"
    with Store(crawl_settings) as store:
        store.save_match(first)
        store.save_match(second)
        with store.conn.cursor() as cursor:
            cursor.execute(
                "SELECT puuid FROM players WHERE lower(game_name)='guardnew'"
                " AND lower(tag_line)='bbbb'"
            )
            assert cursor.fetchall() == []
            cursor.execute(
                "SELECT game_name, tag_line FROM player_names WHERE puuid='puuid-g0'"
                " ORDER BY game_name"
            )
            assert [tuple(row.values()) for row in cursor.fetchall()] == [
                ("guardnew", "bbbb"),
                ("guardold", "aaaa"),
            ]
