import hashlib
import json
import logging

from .config import Settings, get_settings

logger = logging.getLogger(__name__)
PREFIX = "syn"
MODEL_ARTEFACTS = ("synergy_model.pkl", "training_report.json")


class Cache:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = None
        self._build: str | None = None
        try:
            import redis

            client = redis.Redis.from_url(
                self.settings.redis_url, socket_connect_timeout=2, socket_timeout=2
            )
            client.ping()
            self.client = client
        except Exception as exc:
            logger.info("cache unavailable, running without it: %s", exc)

    @property
    def ready(self) -> bool:
        return self.client is not None

    @property
    def build(self) -> str:
        if self._build is None:
            digest = hashlib.sha1()
            for name in MODEL_ARTEFACTS:
                path = self.settings.model_dir / name
                digest.update(str(int(path.stat().st_mtime) if path.exists() else 0).encode())
            self._build = digest.hexdigest()[:10]
        return self._build

    def _key(self, namespace: str, name: str) -> str:
        return f"{PREFIX}:{namespace}:{name}"

    def _versioned(self, namespace: str, name: str) -> str:
        return f"{PREFIX}:{self.build}:{namespace}:{name}"

    def members(self, namespace: str, name: str) -> set[str]:
        if self.client is None:
            return set()
        try:
            return {value.decode() for value in self.client.smembers(self._key(namespace, name))}
        except Exception:
            return set()

    def add_members(self, namespace: str, name: str, values: list[str]) -> None:
        if self.client is None or not values:
            return
        try:
            self.client.sadd(self._key(namespace, name), *values)
        except Exception as exc:
            logger.debug("cache set write failed: %s", exc)

    def shared(self, namespace: str, left: str, right: str) -> set[str]:
        if self.client is None:
            return set()
        try:
            found = self.client.sinter(
                self._key(namespace, left), self._key(namespace, right)
            )
        except Exception:
            return set()
        return {value.decode() for value in found}

    def rank(self, namespace: str, name: str, scores: dict[str, float]) -> None:
        if self.client is None or not scores:
            return
        try:
            self.client.zadd(self._versioned(namespace, name), scores)
        except Exception as exc:
            logger.debug("cache rank write failed: %s", exc)

    def top(self, namespace: str, name: str, limit: int = 10, worst: bool = False):
        if self.client is None:
            return []
        key = self._versioned(namespace, name)
        try:
            found = (
                self.client.zrange(key, 0, limit - 1, withscores=True)
                if worst
                else self.client.zrevrange(key, 0, limit - 1, withscores=True)
            )
        except Exception:
            return []
        return [(member.decode(), score) for member, score in found]

    def get(self, namespace: str, name: str):
        if self.client is None:
            return None
        try:
            raw = self.client.get(self._key(namespace, name))
        except Exception:
            return None
        return json.loads(raw) if raw else None

    def set(self, namespace: str, name: str, payload, ttl: int | None = None) -> None:
        if self.client is None:
            return
        try:
            self.client.set(
                self._key(namespace, name),
                json.dumps(payload),
                ex=ttl if ttl is not None else self.settings.cache_ttl,
            )
        except Exception as exc:
            logger.debug("cache write failed: %s", exc)

    def drop(self, namespace: str) -> int:
        if self.client is None:
            return 0
        removed = 0
        try:
            for key in self.client.scan_iter(f"{PREFIX}:{namespace}:*"):
                removed += int(self.client.delete(key))
        except Exception:
            return removed
        return removed


_cache: Cache | None = None


def get_cache(settings: Settings | None = None) -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(settings)
    return _cache
