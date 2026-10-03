import pytest
from fastapi.testclient import TestClient

from synergy.api import create_app


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def test_status_answers_even_before_a_model_exists(client):
    body = client.get("/api/status").json()
    assert "ready" in body
    assert "cache" in body


def test_endpoints_refuse_politely_when_the_model_is_missing(client):
    for path in ("/api/players", "/api/players/nobody", "/api/partners/nobody"):
        response = client.get(path)
        assert response.status_code in (200, 404, 503)


def test_pair_requires_both_players(client):
    assert client.get("/api/pair", params={"a": "one#tag"}).status_code == 422


def test_a_team_needs_between_two_and_five_players(client):
    assert client.post("/api/team", json={"players": ["only#one"]}).status_code == 422
    assert client.post("/api/team", json={"players": [f"p{i}#t" for i in range(6)]}).status_code == 422


def test_an_outsider_with_no_games_reports_zero_rather_than_failing(client):
    body = client.get("/api/outsider/definitely nobody#zzzz").json()
    assert body["games"] == 0
    assert body["features"] == {}


def test_an_outsider_pair_separates_teammates_from_opponents(client):
    body = client.get("/api/outsider-pair", params={"a": "nobody#a", "b": "nobody#b"}).json()
    assert body["as_teammates"] == []
    assert body["as_opponents"] == []
    assert set(body["as_teammates"]) | set(body["as_opponents"]) <= set(body["shared_matches"])


def test_a_request_fetches_the_model_from_the_store_when_the_service_is_not_ready(monkeypatch):
    # given
    from synergy.api import server
    from synergy.ml import score

    fetched = []

    class _Settings:
        model_store = "s3://models"

    class _NotReady:
        ready = False

    class _Ready:
        ready = True

    answers = [_NotReady(), _Ready()]
    monkeypatch.setattr(server, "get_settings", lambda: _Settings())
    monkeypatch.setattr(server, "fetch_run", lambda settings: fetched.append(settings.model_store))
    monkeypatch.setattr(server, "get_service", lambda: answers.pop(0))
    monkeypatch.setattr(score, "_service", None)

    # when
    service = server._load_service()

    # then
    assert service.ready is True and fetched == ["s3://models"]


def test_a_refresh_passes_the_newest_game_we_hold_and_shapes_the_answer(monkeypatch):
    # given
    import pandas as pd

    from synergy.api import server
    from synergy.api.accounts import LinkError

    asked = []

    class _Service:
        def __init__(self, latest):
            self.latest = latest

        def resolve(self, name):
            return pd.Series({"puuid": "p", "game_name": "Alpha", "tag_line": "NA1", "latest_game": self.latest})

    class _Evaluations:
        def __init__(self):
            self.state = None

        def refresh(self, user, name, latest):
            asked.append((user, name, latest))
            if latest:
                raise LinkError("We already have Alpha#NA1's games from the last two weeks.")
            self.state = {"status": "requested", "riot_id": name, "started": 1}
            raise LinkError("We're pulling Alpha#NA1's newest games now.")

        def lookup(self, name):
            return self.state

    evaluations = _Evaluations()
    monkeypatch.setattr(server, "get_evaluations", lambda: evaluations)

    # when
    monkeypatch.setattr(server, "_ready", lambda: _Service(1_790_000_000))
    refused = server._refresh("user-1", "alpha#na1")
    monkeypatch.setattr(server, "_ready", lambda: _Service(float("nan")))
    started = server._refresh("user-1", "alpha#na1")

    # then
    assert asked == [("user-1", "Alpha#NA1", 1_790_000_000), ("user-1", "Alpha#NA1", None)]
    assert "last two weeks" in refused["message"] and refused["pending"]["status"] == "refused"
    assert "newest games" in started["message"] and started["pending"]["status"] == "requested" and started["pending"]["riot_id"] == "Alpha#NA1"


def _share_server(monkeypatch, version):
    from synergy.api import server
    from synergy.api.shares import Shares

    class _Table:
        def __init__(self):
            self.items = {}

        def get(self, pk, sk):
            found = self.items.get((pk, sk))
            return dict(found) if found else None

        def put(self, pk, sk, values):
            self.items[(pk, sk)] = dict(values)

    class _Verifier:
        def subject(self, token):
            return {"t-owner": "owner", "t-other": "other"}[token]

    class _Accounts:
        def linked_puuid(self, user):
            return "me"

        def status(self, user):
            return {"remaining": 9}

    shares = Shares(_Table(), token=lambda: "tok")
    monkeypatch.setattr(server, "get_shares", lambda: shares)
    monkeypatch.setattr(server, "get_verifier", lambda: _Verifier())
    monkeypatch.setattr(server, "get_accounts", lambda: _Accounts())
    monkeypatch.setattr(server, "_version", lambda puuids: dict(version))
    return server, shares


def test_a_saved_pair_is_served_again_without_a_new_check_and_the_use_is_logged(monkeypatch, caplog):
    # given
    import logging

    version = {"run": "r1", "players": {"me": "", "friend": "ready@1"}}
    server, shares = _share_server(monkeypatch, version)
    checks = []
    found = {"score": 61.0, "interaction": {"score": 61.0}, "players": [{"puuid": "me", "riot_id": "Me#NA1"}, {"puuid": "friend", "riot_id": "Friend#NA1"}], "remaining": 10}

    def check():
        checks.append(1)
        return dict(found)

    # when
    first = server._shared_pair("t-owner", ("pair", "Friend#NA1", None, None), check)
    with caplog.at_level(logging.INFO, logger="synergy.api.server"):
        again = server._shared_pair("t-owner", ("pair", "friend#na1", None, None), check)
        opened = server._open_share("tok", "t-other")
        anonymous = server._open_share("tok", None)

    # then
    assert checks == [1] and first["share"] == again["share"] == "tok" and again["remaining"] == 9 and again["score"] == 61.0
    assert opened["score"] == 61.0 and "remaining" not in opened and anonymous["shared_at"] == opened["shared_at"]
    assert "share served share=tok viewer=owner pair=Me#NA1 + Friend#NA1" in caplog.text
    assert "viewer=account" in caplog.text and "viewer=anonymous" in caplog.text


def test_a_shared_pair_expires_once_new_games_are_pulled_and_the_owner_gets_a_fresh_check(monkeypatch):
    # given
    version = {"run": "r1", "players": {"me": "", "friend": "ready@1"}}
    server, shares = _share_server(monkeypatch, version)
    scores = iter([61.0, 64.0])

    def check():
        score = next(scores)
        return {"score": score, "interaction": {"score": score}, "players": [{"puuid": "me", "riot_id": "Me#NA1"}, {"puuid": "friend", "riot_id": "Friend#NA1"}]}

    server._shared_pair("t-owner", ("pair", "Friend#NA1", None, None), check)

    # when
    version["players"]["friend"] = "requested@2"
    expired = server._open_share("tok", None)
    fresh = server._shared_pair("t-owner", ("pair", "Friend#NA1", None, None), check)
    reopened = server._open_share("tok", None)

    # then
    assert expired["expired"] is True and [player["riot_id"] for player in expired["players"]] == ["Me#NA1", "Friend#NA1"]
    assert fresh["score"] == 64.0 and fresh["share"] == "tok" and reopened["score"] == 64.0 and "expired" not in reopened
