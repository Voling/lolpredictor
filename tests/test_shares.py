from synergy.api.shares import SHARE_DAYS, Shares, pair_puuids, query_key, stamp

START = 1_790_000_000


class _Table:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)


def _found(score):
    return {"score": score, "players": [{"puuid": "me", "riot_id": "Me#NA1"}, {"puuid": "friend", "riot_id": "Friend#NA1"}], "remaining": 7, "share": "old"}


def test_a_pair_keeps_one_link_per_query_and_the_link_carries_the_result_and_its_version():
    # given
    tokens = iter(["first", "second"])
    now = {"t": START}
    shares = Shares(_Table(), clock=lambda: now["t"], token=lambda: next(tokens))
    query = ("pair", "Friend#NA1", "top", "jungle", "me")

    # when
    first = shares.save("user-1", query, _found(55.0), {"run": "r1", "players": {"me": "", "friend": "ready@1"}})
    now["t"] += 60
    again = shares.save("user-1", ("pair", " friend#na1 ", "TOP", "jungle", "me"), _found(57.0), {"run": "r1", "players": {"me": "", "friend": "ready@2"}})
    other = shares.save("user-2", query, _found(50.0), {"run": "r1", "players": {}})
    loaded = shares.load(first)

    # then
    assert first == again == "first" and other == "second"
    assert loaded["payload"] == {"score": 57.0, "players": _found(0)["players"]} and loaded["owner"] == "user-1" and loaded["at"] == START + 60
    assert loaded["version"] == {"run": "r1", "players": {"me": "", "friend": "ready@2"}}
    assert shares.find("user-1", query)["token"] == "first" and pair_puuids(loaded["payload"]) == ["me", "friend"]


def test_a_link_lapses_after_its_days_and_an_unknown_link_finds_nothing():
    # given
    now = {"t": START}
    shares = Shares(_Table(), clock=lambda: now["t"], token=lambda: "tok")
    shares.save("user-1", ("pair", "b", None, None, "me"), _found(55.0), {"run": "r1", "players": {}})

    # when
    now["t"] += SHARE_DAYS * 86400
    lapsed = shares.load("tok")

    # then
    assert lapsed is None and shares.load("missing") is None and shares.find("user-9", ("pair",)) is None


def test_the_version_stamp_changes_when_new_games_are_pulled():
    # given
    ready = {"status": "ready", "started": 100, "finished": 160}
    pulling = {"status": "requested", "started": 900}

    # when
    stamps = [stamp(None), stamp(ready), stamp(pulling), stamp({"status": "ready", "started": 100})]

    # then
    assert stamps == ["", "ready@160", "requested@900", "ready@100"]
    assert query_key(("pair", "A#NA1", None)) == query_key(("PAIR", " a#na1 ", ""))
