import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Path, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..cache import get_cache
from ..config import get_settings
from ..features.outsider import corpus_quantiles, outsider_pair, outsider_profile
from ..ml import score
from ..ml.score import UNKNOWN_NOTE, UnknownPlayer, epoch_of, get_service, reload_service, riot_id
from ..ml.serving import champions_of, warm
from ..pipeline import served_summary
from .accounts import Accounts, AuthError, Busy, DynamoTable, LinkError, QuotaExceeded, RiotAccounts, cognito_verifier, request_key, ssm_key
from .artifacts import fetch_run
from .customs import SharedCustoms
from .evaluate import GIB, Evaluations, shape, store_size
from .history import History, duo_view, public

logger = logging.getLogger(__name__)
LIMIT = 40
POSITION = 12
CHECK_ID = 80
RECENT_CACHE = "public, max-age=30"
_accounts: Accounts | None = None
_customs: SharedCustoms | None = None
_evaluations: Evaluations | None = None
_history: History | None = None
_verifier = None


class TeamRequest(BaseModel):
    players: list[str] = Field(min_length=2, max_length=5)


class LineupRequest(BaseModel):
    players: dict[str, str] = Field(min_length=5, max_length=5)


class LinkRequest(BaseModel):
    riot_id: str = Field(min_length=3, max_length=LIMIT)


class RefreshRequest(BaseModel):
    riot_id: str = Field(min_length=3, max_length=LIMIT)


def get_accounts() -> Accounts:
    global _accounts
    if _accounts is None:
        settings = get_settings()
        key = ssm_key(settings.riot_key_parameter) if settings.riot_key_parameter else (lambda: settings.riot_api_key)
        riot = RiotAccounts(key, settings.platform, settings.region)
        _accounts = Accounts(DynamoTable(settings.accounts_table), riot, settings.daily_duos, settings.link_attempts, settings.riot_budget)
    return _accounts


def get_customs() -> SharedCustoms:
    global _customs
    if _customs is None:
        settings = get_settings()
        accounts = get_accounts()
        _customs = SharedCustoms(accounts.riot, DynamoTable(settings.accounts_table), accounts.riot_calls, settings.target_minute)
    return _customs


def _customs_between(left: str, right: str) -> list[dict]:
    try:
        games, complete = get_customs().between(left, right)
        if not complete:
            logger.info("customs between %s and %s are incomplete this time", left[:8], right[:8])
        return games
    except Exception:
        logger.exception("customs lookup failed")
        return []


def _queue_evaluation(puuid: str) -> None:
    import json

    import boto3

    boto3.client("sqs").send_message(QueueUrl=get_settings().evaluate_queue, MessageBody=json.dumps({"puuid": puuid}))


def get_evaluations() -> Evaluations:
    global _evaluations
    if _evaluations is None:
        settings = get_settings()
        accounts = get_accounts()
        table = DynamoTable(settings.accounts_table)
        _evaluations = Evaluations(
            table,
            accounts.riot,
            accounts.riot_calls,
            launch=_queue_evaluation if settings.evaluate_queue else None,
            store_size=(lambda: store_size(settings, table)) if settings.evaluated_store else (lambda: 0),
            cap_bytes=settings.evaluated_cap_gb * GIB,
            daily=settings.daily_evaluations,
            per_user=settings.user_evaluations,
        )
    return _evaluations


def _pending(user: str, name: str) -> dict | None:
    if not get_settings().evaluate_queue:
        return None
    message = None
    try:
        get_evaluations().refuse(user, name)
    except (LinkError, QuotaExceeded) as exc:
        message = str(exc)
    except Exception:
        logger.exception("could not request an evaluation")
        return None
    return shape(get_evaluations().lookup(name), name, message)


def _refresh(user: str, name: str) -> dict:
    service = _ready()
    latest = None
    try:
        profile = service.resolve(name)
        latest, name = epoch_of(profile.get("latest_game")), riot_id(profile)
    except UnknownPlayer:
        pass
    message = None
    try:
        get_evaluations().refresh(user, name, latest)
    except (LinkError, QuotaExceeded) as exc:
        message = str(exc)
    return {"message": message, "pending": shape(get_evaluations().lookup(name), name, message)}


def get_history() -> History:
    global _history
    if _history is None:
        _history = History(DynamoTable(get_settings().accounts_table))
    return _history


def _save_check(user: str, kind: str, request: str, found: dict, duos: list[dict]) -> None:
    try:
        get_history().save(user, kind, request, found, duos)
    except Exception:
        logger.exception("could not save the check")


def _duos(remember, found: dict, me: str, service) -> list[dict]:
    try:
        return remember(found, me, service)
    except Exception:
        logger.exception("could not describe the duo")
        return []


def get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = cognito_verifier(get_settings())
    return _verifier


def _ready():
    try:
        service = _load_service()
    except Exception:
        logger.exception("could not load the model")
        raise HTTPException(503, "The model is not ready yet. Try again later.")
    if not service.ready:
        raise HTTPException(503, "The model is not ready yet. Try again later.")
    settings = get_settings()
    if getattr(service, "evaluated", None) is None and settings.evaluated_store and settings.accounts_table:
        import boto3

        from ..ml.evaluated import EvaluatedPlayers

        service.evaluated = EvaluatedPlayers(DynamoTable(settings.accounts_table), boto3.client("s3"), settings.evaluated_store.removeprefix("s3://").split("/", 1)[0])
    return service


def _handle(call):
    try:
        return call()
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except QuotaExceeded as exc:
        raise HTTPException(429, str(exc)) from exc
    except Busy as exc:
        raise HTTPException(503, str(exc)) from exc
    except LinkError as exc:
        raise HTTPException(400, str(exc)) from exc
    except UnknownPlayer as exc:
        raise HTTPException(404, f"We can't find {exc} in our data. Check the name and tag.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _metered(token: str | None, query: tuple, count: int, call, unscored=lambda found: 0, remember=lambda found, me, service: []) -> dict:
    user = get_verifier().subject(token)
    accounts = get_accounts()
    me = accounts.linked_puuid(user)
    service = _ready()
    if me not in service.profiles.index and service.adopt_puuid(me) is None:
        pending = _pending(user, accounts.link(user).get("riot_id") or me)
        if pending:
            return {"score": None, "pending": pending, "remaining": accounts.status(user)["remaining"]}
        raise LinkError("Your Riot account isn't in our data yet.")
    request = request_key(*query[:1], accounts.day(), *query[1:])
    left, charged = accounts.spend(user, count, request)
    try:
        found = call(me, service)
    except UnknownPlayer as exc:
        if charged:
            accounts.refund(user, count, request)
        pending = _pending(user, str(exc))
        if pending:
            return {"score": None, "pending": pending, "remaining": accounts.status(user)["remaining"]}
        raise
    except Exception:
        if charged:
            accounts.refund(user, count, request)
        raise
    if get_settings().evaluate_queue:
        for row in found.get("friends", []):
            if row.get("note") == UNKNOWN_NOTE and row.get("riot_id"):
                pending = _pending(user, row["riot_id"])
                if pending:
                    row["pending"] = pending
                    row["note"] = pending.get("message") or row["note"]
    duos = _duos(remember, found, me, service)
    _save_check(user, query[0], request, found, duos)
    back = unscored(found) if charged else 0
    if back:
        accounts.refund(user, back)
    if charged and duos:
        try:
            accounts.remember([public(duo) for duo in duos])
        except Exception:
            logger.exception("could not remember the duo")
    return {**found, "remaining": left + back}


def seat_memory(profile, position: str, prefix: str, settings) -> dict:
    return {
        f"{prefix}_name": riot_id(profile),
        f"{prefix}_champions": ",".join(champions_of(profile["puuid"], position, settings)) or None,
        f"{prefix}_tier": profile.get("tier"),
        f"{prefix}_division": profile.get("division"),
        f"{prefix}_position": position,
    }


def _memory(me_profile, me_position: str, friend_profile, friend_position: str, score: float, gold: float, minute: int) -> dict:
    settings = get_settings()
    return {
        **seat_memory(me_profile, me_position, "left", settings),
        **seat_memory(friend_profile, friend_position, "right", settings),
        "score": round(float(score), 1),
        "gold": round(float(gold), 1),
        "minute": int(minute),
    }


def _remember_pair(found: dict, me: str, service) -> list[dict]:
    interaction = found.get("interaction")
    if not interaction:
        return []
    friend = service.resolve(found["players"][1]["riot_id"])
    return [_memory(service.profiles.loc[me], interaction["positions"]["left"], friend, interaction["positions"]["right"], interaction["score"], interaction["edge"]["total"], interaction["minute"])]


def _remember_friends(found: dict, me: str, service) -> list[dict]:
    own = found["me"]
    return [
        _memory(service.profiles.loc[me], own["position"], service.resolve(row["riot_id"]), row["position"], row["score"], row["total"], own["minute"])
        for row in found["friends"]
        if row.get("score") is not None
    ]


def _public_routes(app: FastAPI) -> None:
    @app.get("/api/me")
    def me(x_auth: str | None = Header(None)):
        return _handle(lambda: get_accounts().status(get_verifier().subject(x_auth)))

    @app.get("/api/evaluations")
    def evaluations(names: str = Query("", max_length=LIMIT * 12), x_auth: str | None = Header(None)):
        def read():
            get_verifier().subject(x_auth)
            wanted = [name.strip() for name in names.split(",") if name.strip()][: get_settings().max_friends + 1]
            return {"players": [shape(get_evaluations().lookup(name), name) for name in wanted]}

        return _handle(read)

    @app.get("/api/history")
    def history(x_auth: str | None = Header(None)):
        return _handle(lambda: {"checks": get_history().list(get_verifier().subject(x_auth))})

    @app.get("/api/history/{check_id}")
    def saved_check(check_id: str = Path(max_length=CHECK_ID), x_auth: str | None = Header(None)):
        def read():
            found = get_history().load(get_verifier().subject(x_auth), check_id)
            if found is None:
                raise HTTPException(404, "That check isn't saved any more.")
            return found

        return _handle(read)

    @app.post("/api/refresh")
    def refresh(request: RefreshRequest, x_auth: str | None = Header(None)):
        return _handle(lambda: _refresh(get_verifier().subject(x_auth), request.riot_id))

    @app.get("/api/recent")
    def recent(response: Response):
        response.headers["Cache-Control"] = RECENT_CACHE
        return _handle(lambda: {"duos": [duo_view(duo) for duo in get_accounts().recent()]})

    @app.post("/api/link")
    def link(request: LinkRequest, x_auth: str | None = Header(None)):
        return _handle(lambda: get_accounts().start_link(get_verifier().subject(x_auth), request.riot_id))

    @app.post("/api/link/verify")
    def verify(x_auth: str | None = Header(None)):
        return _handle(lambda: get_accounts().verify_link(get_verifier().subject(x_auth)))

    @app.get("/api/pair")
    def pair(
        b: str = Query(max_length=LIMIT),
        a_position: str | None = Query(None, max_length=POSITION),
        b_position: str | None = Query(None, max_length=POSITION),
        x_auth: str | None = Header(None),
    ):
        query = ("pair", b, a_position, b_position)
        return _handle(lambda: _metered(x_auth, query, 1, lambda me, service: service.pair_score(me, b, a_position, b_position, details=False, customs=_customs_between), remember=_remember_pair))

    @app.get("/api/friends")
    def friends(
        friends: list[str] = Query(...),
        me_position: str | None = Query(None, max_length=POSITION),
        x_auth: str | None = Header(None),
    ):
        most = get_settings().max_friends
        if len(friends) > most or any(len(friend) > LIMIT for friend in friends):
            raise HTTPException(400, f"Rank up to {most} friends at a time.")
        query = ("friends", me_position, *sorted(" ".join(friend.lower().split()) for friend in friends))
        return _handle(
            lambda: _metered(
                x_auth,
                query,
                len(friends),
                lambda me, service: service.friends(me, friends, me_position, details=False, customs=_customs_between),
                lambda found: sum(1 for row in found["friends"] if row.get("score") is None),
                remember=_remember_friends,
            )
        )


def _local_routes(app: FastAPI) -> None:
    @app.post("/api/reload")
    def reload():
        get_cache().drop("quantiles")
        get_cache().drop("outsider")
        return reload_service().status()

    @app.get("/api/players")
    def players(q: str = Query("", max_length=32), limit: int = Query(25, ge=1, le=200)):
        return {"players": _ready().search(q, limit)}

    @app.get("/api/players/{query}")
    def player(query: str):
        service = _ready()
        return _handle(lambda: service.player_report(query))

    @app.get("/api/partners/{query}")
    def partners(query: str, limit: int = Query(10, ge=1, le=50)):
        service = _ready()
        return _handle(lambda: service.best_partners(query, limit=limit))

    @app.get("/api/pair")
    def pair(a: str, b: str, a_position: str | None = None, b_position: str | None = None):
        service = _ready()
        return _handle(lambda: service.pair_score(a, b, a_position, b_position))

    @app.get("/api/friends")
    def friends(me: str, friends: list[str] = Query(...), me_position: str | None = None):
        service = _ready()
        return _handle(lambda: service.friends(me, friends, me_position))

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


def _load_service():
    with score._service_lock:
        service = get_service()
        if service.ready:
            return service
        settings = get_settings()
        if settings.model_store:
            fetch_run(settings)
        return get_service()


def _warm_up():
    clock = time.time()
    try:
        _load_service()
        warm(get_settings())
        logger.info("warm up done in %.1fs", time.time() - clock)
    except Exception:
        logger.exception("warm up failed")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if app.state.public:
        threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


def create_app(public: bool | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    public = get_settings().public_api if public is None else public
    app = FastAPI(
        title="lolpredictor",
        version="1.0",
        docs_url=None if public else "/docs",
        redoc_url=None,
        openapi_url=None if public else "/openapi.json",
        lifespan=_lifespan,
    )
    app.state.public = public
    origins = [origin for origin in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if origin]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "x-auth"])

    @app.get("/ready")
    def ready():
        return {"ok": True}

    if public:

        @app.get("/api/status")
        def signed_status(x_auth: str | None = Header(None)):
            _handle(lambda: get_verifier().subject(x_auth))
            ready = _handle(_load_service).ready
            run = served_summary()
            return {"ready": ready, "run": {"id": run["id"]} if run else None}

    else:

        @app.get("/api/health")
        def health():
            return {"ok": True}

        @app.get("/api/recent")
        def recent():
            return {"duos": []}

        @app.get("/api/status")
        def status():
            return {**get_service().status(), "cache": get_cache().ready, "run": served_summary()}

    if public:
        _public_routes(app)
    else:
        _local_routes(app)
    return app


app = create_app()
