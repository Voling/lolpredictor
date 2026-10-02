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
