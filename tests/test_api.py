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
