import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from synergy.ml.score import UnknownPlayer, get_service, reload_service

logger = logging.getLogger(__name__)


def _error(message: str, status: int) -> JsonResponse:
    return JsonResponse({"error": message}, status=status)


def _guard(handler):
    def wrapped(request, *args, **kwargs):
        service = get_service()
        if not service.ready:
            return _error("model not trained, run `python -m synergy all` first", 503)
        try:
            return handler(request, service, *args, **kwargs)
        except UnknownPlayer as exc:
            return _error(f"unknown player: {exc}", 404)
        except ValueError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            logger.exception("request failed")
            return _error(str(exc), 500)

    wrapped.__name__ = handler.__name__
    return wrapped


@require_http_methods(["GET"])
def status(request):
    return JsonResponse(get_service().status())


@require_http_methods(["POST"])
@csrf_exempt
def reload(request):
    return JsonResponse(reload_service().status())


@require_http_methods(["GET"])
@_guard
def players(request, service):
    term = request.GET.get("q", "")
    limit = min(int(request.GET.get("limit", 25)), 200)
    return JsonResponse({"players": service.search(term, limit)})


@require_http_methods(["GET"])
@_guard
def player(request, service, query):
    return JsonResponse(service.player_report(query))


@require_http_methods(["GET"])
@_guard
def partners(request, service, query):
    limit = min(int(request.GET.get("limit", 10)), 50)
    return JsonResponse(service.best_partners(query, limit=limit))


@require_http_methods(["GET"])
@_guard
def pair(request, service):
    left = request.GET.get("a")
    right = request.GET.get("b")
    if not left or not right:
        raise ValueError("both a and b query parameters are required")
    return JsonResponse(service.pair_score(left, right))


@csrf_exempt
@require_http_methods(["POST"])
@_guard
def team(request, service):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        raise ValueError("invalid json body")
    queries = [str(item).strip() for item in payload.get("players", []) if str(item).strip()]
    if len(queries) < 2:
        raise ValueError("provide at least two players")
    if len(queries) > 5:
        raise ValueError("a team holds at most five players")
    return JsonResponse(service.team_report(queries))
