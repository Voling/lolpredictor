import hashlib
import json
import secrets
import time

from .accounts import DAY
from .history import _plain

SHARE = "share"
SHARES = "shares"
SHARE_DAYS = 30
TOKEN_BYTES = 9


def query_key(query: tuple) -> str:
    text = "|".join(" ".join(str(part or "").lower().split()) for part in query)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def stamp(state: dict | None) -> str:
    if not state:
        return ""
    return f"{state.get('status')}@{int(state.get('finished') or state.get('started') or 0)}"


def pair_puuids(payload: dict) -> list[str]:
    return [str(player["puuid"]) for player in payload.get("players", [])]


class Shares:
    def __init__(self, table, clock=time.time, token=lambda: secrets.token_urlsafe(TOKEN_BYTES)):
        self.table, self.clock, self.token = table, clock, token

    def _mine(self, user: str) -> str:
        return f"{SHARES}#{user}"

    def find(self, user: str, query: tuple) -> dict | None:
        mapping = self.table.get(self._mine(user), query_key(query))
        return None if mapping is None else self.load(mapping["token"])

    def save(self, user: str, query: tuple, found: dict, version: dict) -> str:
        key = query_key(query)
        mapping = self.table.get(self._mine(user), key)
        token = mapping["token"] if mapping else self.token()
        now = int(self.clock())
        expires = now + SHARE_DAYS * DAY
        kept = {name: value for name, value in found.items() if name not in ("remaining", "share")}
        self.table.put(
            SHARE,
            token,
            {"owner": user, "payload": json.dumps(kept, default=_plain), "version": json.dumps(version, sort_keys=True), "at": now, "expires": expires},
        )
        self.table.put(self._mine(user), key, {"token": token, "expires": expires})
        return token

    def load(self, token: str) -> dict | None:
        item = self.table.get(SHARE, token)
        if item is None or int(item.get("expires", 0)) <= int(self.clock()):
            return None
        return {"token": token, "owner": item["owner"], "payload": json.loads(item["payload"]), "version": json.loads(item["version"]), "at": int(item["at"])}
