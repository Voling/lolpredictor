from synergy.api.history import HISTORY_DAYS, History, duo_view, public

START = 1_790_000_000


class _Table:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)

    def latest(self, pk, limit, fields=None):
        keys = sorted((key for key in self.items if key[0] == pk), key=lambda key: key[1], reverse=True)[:limit]
        return [{"sk": key[1], **{name: value for name, value in self.items[key].items() if not fields or name in fields}} for key in keys]


def _pair(score):
    return {"score": score, "positions": {"left": "TOP", "right": "JUNGLE"}, "players": [{"riot_id": "Me#NA1"}, {"riot_id": "Friend#NA1"}], "remaining": 7}


def test_a_check_is_kept_once_per_query_and_day_and_listed_newest_first():
    # given
    now = {"t": START}
    history = History(_Table(), clock=lambda: now["t"])
    friends = {"me": {"riot_id": "Me#NA1", "position": "TOP"}, "friends": [{"riot_id": "A#NA1", "position": "JUNGLE", "score": 55.0}, {"riot_id": "B#NA1", "note": "Not in our data yet."}]}

    # when
    history.save("user-1", "pair", "abc", _pair(61.0))
    now["t"] += 60
    history.save("user-1", "pair", "abc", _pair(62.0))
    now["t"] += 60
    history.save("user-1", "friends", "def", friends)
    now["t"] += 2 * 86400
    history.save("user-1", "pair", "abc", _pair(63.0))
    listed = history.list("user-1")

    # then
    assert [(row["kind"], row["score"], row["names"]) for row in listed] == [
        ("pair", 63.0, ["Me#NA1", "Friend#NA1"]),
        ("friends", 55.0, ["Me#NA1", "A#NA1", "B#NA1"]),
        ("pair", 62.0, ["Me#NA1", "Friend#NA1"]),
    ]
    assert listed[1]["positions"] == ["TOP", "JUNGLE", None] and listed[0]["at"] == START + 120 + 2 * 86400


def test_a_saved_check_loads_without_the_quota_and_is_kept_ninety_days():
    # given
    table = _Table()
    history = History(table, clock=lambda: START)

    # when
    check_id = history.save("user-1", "pair", "abc", _pair(61.0))
    found = history.load("user-1", check_id)
    item = table.items[("checks#user-1", check_id)]

    # then
    assert check_id.endswith("#abc") and len(check_id) == 14
    assert found["score"] == 61.0 and found["saved_at"] == START and found["kind"] == "pair" and "remaining" not in found
    assert item["expires"] == START + HISTORY_DAYS * 86400
    assert history.load("user-2", check_id) is None and history.load("user-1", "nope") is None


def _duo(score):
    return {
        "left_name": "Me#NA1", "left_champions": "Ezreal,Caitlyn,Sivir", "left_tier": "MASTER", "left_position": "BOTTOM",
        "right_name": "Friend#NA1", "right_champions": "Aurora,Akali,Ahri", "right_tier": "DIAMOND", "right_division": "I", "right_position": "MIDDLE",
        "score": score, "gold": 120.0, "minute": 20,
    }


def test_a_check_keeps_its_duos_with_names_and_lists_them_as_seats():
    # given
    history = History(_Table(), clock=lambda: START)

    # when
    history.save("user-1", "pair", "abc", _pair(61.0), [_duo(61.0)])
    listed = history.list("user-1")

    # then
    duo = listed[0]["duos"][0]
    assert duo["left"] == {"name": "Me#NA1", "champions": ["Ezreal", "Caitlyn", "Sivir"], "tier": "MASTER", "division": None, "position": "BOTTOM"}
    assert duo["right"]["champions"] == ["Aurora", "Akali", "Ahri"] and duo["right"]["name"] == "Friend#NA1"
    assert duo["score"] == 61.0 and duo["at"] == START and listed[0]["names"] == ["Me#NA1", "Friend#NA1"]


def test_the_public_list_drops_names_and_an_old_single_champion_is_not_shown():
    # given
    old = {"left_champion": "Pantheon", "left_tier": "MASTER", "left_position": "BOTTOM", "right_champion": "Akali", "right_tier": "DIAMOND", "right_position": "MIDDLE", "score": 51.4, "gold": 80.0, "minute": 20, "at": START}

    # when
    shared = public(_duo(55.0))
    legacy = duo_view(old)

    # then
    assert "left_name" not in shared and "right_name" not in shared and shared["left_champions"] == "Ezreal,Caitlyn,Sivir"
    assert legacy["left"] == {"champions": [], "tier": "MASTER", "division": None, "position": "BOTTOM"} and legacy["score"] == 51.4
