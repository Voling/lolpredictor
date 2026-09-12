from synergy.cache import Cache
from synergy.config import Settings


def unreachable() -> Cache:
    return Cache(Settings(redis_url="redis://127.0.0.1:65344/0"))


def test_a_missing_server_degrades_to_no_cache():
    cache = unreachable()
    assert not cache.ready
    assert cache.get("ns", "key") is None
    cache.set("ns", "key", {"a": 1})
    assert cache.get("ns", "key") is None
    assert cache.drop("ns") == 0


def test_values_round_trip_when_a_server_is_present():
    cache = Cache()
    if not cache.ready:
        return
    cache.drop("unit")
    cache.set("unit", "thing", {"a": 1, "b": [2, 3]}, ttl=60)
    assert cache.get("unit", "thing") == {"a": 1, "b": [2, 3]}
    assert cache.drop("unit") == 1
    assert cache.get("unit", "thing") is None
