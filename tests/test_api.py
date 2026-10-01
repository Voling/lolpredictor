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
