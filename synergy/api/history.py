import json
import time
from datetime import datetime, timezone

from .accounts import DAY

CHECKS = "checks"
HISTORY_DAYS = 90
HISTORY_LIMIT = 50
LISTED = ("at", "kind", "summary")
SIDES = ("left", "right")
SEAT_FIELDS = ("tier", "division", "position")


def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)


def public(duo: dict) -> dict:
    return {name: value for name, value in duo.items() if not name.endswith("_name")}


def _champions(duo: dict, side: str) -> list[str]:
    found = duo.get(f"{side}_champions")
    return [name for name in str(found).split(",") if name] if found else []


def _seat_view(duo: dict, side: str) -> dict:
    view = {"champions": _champions(duo, side), **{name: duo.get(f"{side}_{name}") for name in SEAT_FIELDS}}
    if duo.get(f"{side}_name"):
        view["name"] = duo[f"{side}_name"]
    return view


def duo_view(duo: dict) -> dict:
    return {**{side: _seat_view(duo, side) for side in SIDES}, **{name: duo.get(name) for name in ("score", "gold", "minute", "at")}}


def summary_of(kind: str, found: dict) -> dict:
    if kind == "pair":
        positions = found.get("positions") or {}
        return {
            "names": [player["riot_id"] for player in found.get("players", [])],
            "positions": [positions.get("left"), positions.get("right")],
            "score": found.get("score"),
        }
    rows = found.get("friends", [])
    me = found.get("me", {})
    scores = [row["score"] for row in rows if row.get("score") is not None]
    return {
        "names": [me.get("riot_id"), *[row["riot_id"] for row in rows]],
        "positions": [me.get("position"), *[row.get("position") for row in rows]],
        "score": max(scores) if scores else None,
    }


class History:
    def __init__(self, table, clock=time.time):
        self.table, self.clock = table, clock

    def _key(self, user: str) -> str:
        return f"{CHECKS}#{user}"

    def save(self, user: str, kind: str, request: str, found: dict, duos: list[dict] | tuple = ()) -> str:
        now = int(self.clock())
        check_id = f"{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d}#{request}"
        kept = {name: value for name, value in found.items() if name != "remaining"}
        self.table.put(
            self._key(user),
            check_id,
            {
                "at": now,
                "kind": kind,
                "summary": json.dumps({**summary_of(kind, found), "duos": list(duos)}, default=_plain),
                "payload": json.dumps(kept, default=_plain),
                "expires": now + HISTORY_DAYS * DAY,
            },
        )
        return check_id

    def list(self, user: str, limit: int = HISTORY_LIMIT) -> list[dict]:
        rows = []
        for item in self.table.latest(self._key(user), limit, LISTED):
            summary, at = json.loads(item["summary"]), int(item["at"])
            duos = [duo_view({**duo, "at": at}) for duo in summary.pop("duos", [])]
            rows.append({"id": item["sk"], "at": at, "kind": item["kind"], **summary, "duos": duos})
        return sorted(rows, key=lambda row: -row["at"])

    def load(self, user: str, check_id: str) -> dict | None:
        item = self.table.get(self._key(user), check_id)
        if item is None:
            return None
        return {**json.loads(item["payload"]), "kind": item["kind"], "saved_at": int(item["at"])}
