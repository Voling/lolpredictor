import pytest

from synergy.api.accounts import LinkError, QuotaExceeded
from synergy.api.evaluate import EVAL, FAILED, NAMES, READY, Evaluations, shape

START = 1_790_000_000.0


class _Table:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)

    def add(self, pk, sk, amount, limit, expires):
        found = self.items.get((pk, sk))
        if found is not None and found["used"] > limit - amount:
            return None
        used = (found or {"used": 0})["used"] + amount
        self.items[(pk, sk)] = {"used": used, "expires": expires}
        return used


class _Riot:
    def __init__(self):
        self.calls = 0

    def account(self, riot_id):
        self.calls += 1
        name, _, tag = riot_id.partition("#")
        if name == "ghost":
            raise LinkError("Riot doesn't know that Riot ID. Check the name and tag.")
        return {"puuid": f"puuid-{name.lower()}", "gameName": name, "tagLine": tag.upper()}


def _evaluations(launched=None, clock=lambda: START, size=0, daily=2, per_user=1):
    launched = [] if launched is None else launched
    return Evaluations(_Table(), _Riot(), lambda calls: True, launch=launched.append, store_size=lambda: size, daily=daily, per_user=per_user, clock=clock)


def test_an_unknown_player_is_requested_once_and_the_task_is_started():
    # given
    launched = []
    now = {"t": START}
    evaluations = _evaluations(launched, clock=lambda: now["t"])

    # when
    with pytest.raises(LinkError) as first:
        evaluations.refuse("user-1", " thiqums#crocs ")
    now["t"] += 60
    with pytest.raises(LinkError) as again:
        evaluations.refuse("user-2", "Thiqums#CROCS")

    # then
    assert launched == ["puuid-thiqums"] and evaluations.riot.calls == 1
    assert "pulling their last 50 ranked games" in str(first.value) and "still pulling thiqums#CROCS's games" in str(again.value)
    assert evaluations.lookup("thiqums#crocs")["status"] == "requested" and evaluations.table.get(NAMES, "thiqums#crocs")["puuid"] == "puuid-thiqums"


def test_each_user_and_the_whole_site_have_a_daily_allowance_of_new_players():
    # given
    evaluations = _evaluations()

    # when
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "a#na1")
    with pytest.raises(QuotaExceeded) as personal:
        evaluations.refuse("user-1", "b#na1")
    with pytest.raises(LinkError):
        evaluations.refuse("user-2", "b#na1")
    with pytest.raises(LinkError) as everyone:
        evaluations.refuse("user-3", "c#na1")

    # then
    assert "1 new players a day" in str(personal.value) and "as many new players as we can today" in str(everyone.value)
    assert evaluations.table.get("user#user-3", f"evals#{evaluations.day()}")["used"] == 0


def test_a_full_store_refuses_before_spending_anything():
    # given
    launched = []
    evaluations = _evaluations(launched, size=10 * 1024**3)

    # when
    with pytest.raises(LinkError) as full:
        evaluations.refuse("user-1", "a#na1")

    # then
    assert "store is full" in str(full.value) and launched == [] and evaluations.riot.calls == 0


def test_a_stale_request_is_retried_and_ready_or_failed_states_explain_themselves():
    # given
    launched = []
    now = {"t": START}
    evaluations = _evaluations(launched, clock=lambda: now["t"], daily=10, per_user=10)
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "a#na1")

    # when
    now["t"] += 1801
    with pytest.raises(LinkError) as retried:
        evaluations.refuse("user-1", "a#na1")
    evaluations.table.put(EVAL, "puuid-a", {"status": READY, "riot_id": "a#NA1", "started": int(now["t"]), "expires": int(now["t"]) + 100})
    with pytest.raises(LinkError) as ready:
        evaluations.refuse("user-1", "a#na1")
    evaluations.table.put(EVAL, "puuid-a", {"status": FAILED, "riot_id": "a#NA1", "started": int(now["t"]), "expires": int(now["t"]) + 100})
    with pytest.raises(LinkError) as failed:
        evaluations.refuse("user-1", "a#na1")

    # then
    assert launched == ["puuid-a", "puuid-a"] and "pulling their last" in str(retried.value)
    assert "hasn't picked them up yet" in str(ready.value) and "couldn't pull a#NA1's games" in str(failed.value)


def test_a_launch_that_fails_marks_the_request_failed():
    # given
    def broken(puuid):
        raise RuntimeError("no cluster")

    evaluations = Evaluations(_Table(), _Riot(), lambda calls: True, launch=broken, clock=lambda: START)

    # when
    with pytest.raises(LinkError) as refused:
        evaluations.refuse("user-1", "a#na1")

    # then
    assert "couldn't start pulling" in str(refused.value) and evaluations.status("puuid-a")["status"] == FAILED


def test_a_refused_request_without_a_state_is_shaped_as_not_started_with_the_reason():
    # given
    reason = "You can ask for 3 new players a day. Try again tomorrow."

    # when
    refused = shape(None, "GodsGlory7#NA1", reason)
    queued = shape({"status": "requested", "riot_id": "GodsGlory7#NA1"}, "godsglory7#na1")

    # then
    assert refused == {"riot_id": "GodsGlory7#NA1", "status": "refused", "step": "refused", "done": 0, "total": 0, "percent": 0, "message": reason}
    assert queued["status"] == queued["step"] == "requested" and queued["riot_id"] == "GodsGlory7#NA1" and queued["message"] is None


def test_a_name_riot_does_not_know_gives_the_daily_slots_back():
    # given
    launched = []
    evaluations = _evaluations(launched, daily=2, per_user=1)

    # when
    with pytest.raises(LinkError) as unknown:
        evaluations.refuse("user-1", "ghost#NA1")
    with pytest.raises(LinkError) as started:
        evaluations.refuse("user-1", "thiqums#crocs")

    # then
    assert "doesn't know" in str(unknown.value) and "pulling" in str(started.value) and launched == ["puuid-thiqums"]
    assert evaluations.table.items[("user#user-1", f"evals#{evaluations.day()}")]["used"] == 1
    assert evaluations.table.items[(EVAL, f"day#{evaluations.day()}")]["used"] == 1
