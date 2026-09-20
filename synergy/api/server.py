import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..cache import get_cache
from ..features.outsider import corpus_quantiles, outsider_pair, outsider_profile
from ..ml.score import UnknownPlayer, get_service, reload_service
from ..pipeline import served_summary

logger = logging.getLogger(__name__)


class TeamRequest(BaseModel):
    players: list[str] = Field(min_length=2, max_length=5)


class LineupRequest(BaseModel):
    players: dict[str, str] = Field(min_length=5, max_length=5)


def _ready():
    service = get_service()
    if not service.ready:
        raise HTTPException(503, "model not trained, run `python -m synergy all` first")
    return service


def _handle(call):
    try:
        return call()
    except UnknownPlayer as exc:
        raise HTTPException(404, f"unknown player: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def create_app() -> FastAPI:
    app = FastAPI(title="lolpredictor", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status():
        service = get_service()
        return {**service.status(), "cache": get_cache().ready, "run": served_summary()}

    @app.post("/api/reload")
    def reload():
        get_cache().drop("quantiles")
        get_cache().drop("outsider")
        return reload_service().status()

    @app.get("/api/players")
    def players(q: str = "", limit: int = Query(25, le=200)):
        return {"players": _ready().search(q, limit)}

    @app.get("/api/players/{query}")
    def player(query: str):
        service = _ready()
        return _handle(lambda: service.player_report(query))

    @app.get("/api/partners/{query}")
    def partners(query: str, limit: int = Query(10, le=50)):
        service = _ready()
        return _handle(lambda: service.best_partners(query, limit=limit))

    @app.get("/api/pair")
    def pair(a: str, b: str, a_position: str | None = None, b_position: str | None = None):
        service = _ready()
        return _handle(lambda: service.pair_score(a, b, a_position, b_position))

    @app.post("/api/team")
    def team(request: TeamRequest):
        service = _ready()
        return _handle(lambda: service.team_report(request.players))

    @app.post("/api/lineup")
    def lineup(request: LineupRequest):
        service = _ready()
        return _handle(lambda: service.lineup(request.players))

    @app.get("/api/outsider/{riot_id}")
    def outsider(riot_id: str, refresh: bool = False):
        return _handle(lambda: outsider_profile(riot_id, refresh=refresh))

    @app.get("/api/outsider-pair")
    def outsiders(a: str, b: str):
        return _handle(lambda: outsider_pair(a, b))

    @app.get("/api/corpus")
    def corpus():
        payload = corpus_quantiles()
        return {"rows": payload["rows"], "columns": payload["columns"]}

    return app


app = create_app()
