import json
import time
from datetime import datetime, timezone

from .accounts import DAY

CHECKS = "checks"
HISTORY_DAYS = 90
HISTORY_LIMIT = 50
LISTED = ("at", "kind", "summary")


def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)


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

    def save(self, user: str, kind: str, request: str, found: dict) -> str:
        now = int(self.clock())
        check_id = f"{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d}#{request}"
        kept = {name: value for name, value in found.items() if name != "remaining"}
        self.table.put(
            self._key(user),
            check_id,
            {
                "at": now,
                "kind": kind,
                "summary": json.dumps(summary_of(kind, found), default=_plain),
                "payload": json.dumps(kept, default=_plain),
                "expires": now + HISTORY_DAYS * DAY,
            },
        )
        return check_id

    def list(self, user: str, limit: int = HISTORY_LIMIT) -> list[dict]:
        rows = [
            {"id": item["sk"], "at": int(item["at"]), "kind": item["kind"], **json.loads(item["summary"])}
            for item in self.table.latest(self._key(user), limit, LISTED)
        ]
        return sorted(rows, key=lambda row: -row["at"])

    def load(self, user: str, check_id: str) -> dict | None:
        item = self.table.get(self._key(user), check_id)
        if item is None:
            return None
        return {**json.loads(item["payload"]), "kind": item["kind"], "saved_at": int(item["at"])}
