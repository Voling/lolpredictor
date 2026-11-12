import pytest

from synergy.api.accounts import RIOT_DOWN, LinkError, LocalTable, QuotaExceeded, RateLimited, RiotBusy, RiotUnavailable
from synergy.api.evaluate import EVAL, FAILED, NAMES, PLAYER, READY, REQUESTED, SYSTEM, SYSTEM_PROBLEM, Evaluations, Progress, fail_evaluation, shape

START = 1_790_000_000.0


class _Riot:
    def __init__(self, busy=False, down=False):
        self.calls, self.busy, self.down = 0, busy, down

    def account(self, riot_id):
        if self.down:
            raise RiotUnavailable(420.0)
        if self.busy:
            raise RateLimited(30.0)
        self.calls += 1
        name, _, tag = riot_id.partition("#")
        if name == "ghost":
            raise LinkError("Riot doesn't know that Riot ID. Check the name and tag.")
        return {"puuid": f"puuid-{name.lower()}", "gameName": name, "tagLine": tag.upper()}


def _evaluations(launched=None, clock=lambda: START, size=0, daily=2, per_user=1):
    launched = [] if launched is None else launched
    return Evaluations(LocalTable(), _Riot(), launch=launched.append, store_size=lambda: size, daily=daily, per_user=per_user, clock=clock)


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

    evaluations = Evaluations(LocalTable(), _Riot(), launch=broken, clock=lambda: START)

    # when
    with pytest.raises(LinkError) as refused:
        evaluations.refuse("user-1", "a#na1")

    # then
    state = evaluations.status("puuid-a")
    assert "couldn't start pulling a#NA1's games" in str(refused.value) and (state["status"], state["kind"], state["refunded"]) == (FAILED, SYSTEM, 1)
    assert evaluations.table.get("user#user-1", f"evals#{evaluations.day()}")["used"] == 0 and evaluations.table.get(EVAL, f"day#{evaluations.day()}")["used"] == 0


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


def test_a_refresh_waits_two_weeks_after_the_newest_game_we_hold_then_pulls_again():
    # given
    launched = []
    now = {"t": START}
    evaluations = _evaluations(launched, clock=lambda: now["t"], daily=10, per_user=10)
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "thiqums#crocs")
    evaluations.table.put(EVAL, "puuid-thiqums", {"status": READY, "riot_id": "Thiqums#CROCS", "started": START, "expires": START + 14 * 86400})

    # when
    with pytest.raises(LinkError) as fresh:
        evaluations.refresh("user-1", "thiqums#crocs", latest_game=int(START) - 3 * 86400)
    with pytest.raises(LinkError) as pulling:
        evaluations.refresh("user-1", "thiqums#crocs", latest_game=int(START) - 15 * 86400)
    state = evaluations.status("puuid-thiqums")
    with pytest.raises(LinkError) as busy:
        evaluations.refresh("user-1", "thiqums#crocs")

    # then
    assert "from the last two weeks" in str(fresh.value) and "after October 2" in str(fresh.value)
    assert "newest games" in str(pulling.value) and launched == ["puuid-thiqums", "puuid-thiqums"]
    assert state["status"] == "requested" and state["started"] == int(START) and "still pulling" in str(busy.value)


def test_visitors_have_their_own_daily_allowance_of_new_players_apart_from_accounts():
    # given
    launched = []
    evaluations = Evaluations(LocalTable(), _Riot(), launch=launched.append, daily=1, per_user=1, per_visitor=1, visitors_daily=2, clock=lambda: START)

    # when
    with pytest.raises(LinkError):
        evaluations.refuse("visitor#one", "a#na1")
    with pytest.raises(QuotaExceeded) as personal:
        evaluations.refuse("visitor#one", "b#na1")
    with pytest.raises(LinkError):
        evaluations.refuse("visitor#two", "b#na1")
    with pytest.raises(LinkError) as everyone:
        evaluations.refuse("visitor#three", "c#na1")
    with pytest.raises(LinkError) as account:
        evaluations.refuse("user-1", "c#na1")

    # then
    assert launched == ["puuid-a", "puuid-b", "puuid-c"]
    assert "1 new players a day. Sign in for more." in str(personal.value)
    assert "as many new players as we can today. Sign in for more." in str(everyone.value) and "pulling their last" in str(account.value)


def test_a_visitor_spends_a_new_player_only_when_one_is_queued():
    # given
    launched = []
    evaluations = Evaluations(LocalTable(), _Riot(), launch=launched.append, per_visitor=1, visitors_daily=5, clock=lambda: START)
    with pytest.raises(LinkError):
        evaluations.refuse("visitor#one", "thiqums#crocs")

    def broken(puuid):
        raise RuntimeError("no queue")

    # when
    with pytest.raises(LinkError) as pending:
        evaluations.refuse("visitor#two", "thiqums#euw")
    evaluations.launch = broken
    with pytest.raises(LinkError) as failed:
        evaluations.refuse("visitor#two", "other#na1")

    # then
    assert launched == ["puuid-thiqums"] and "still pulling thiqums#CROCS's games" in str(pending.value) and "couldn't start pulling" in str(failed.value)
    assert evaluations.table.get("visitor#two", f"evals#{evaluations.day()}")["used"] == 0
    assert evaluations.table.get(EVAL, f"visitors#{evaluations.day()}")["used"] == 1


def test_a_visitor_asks_riot_through_the_visitor_budget_and_a_busy_one_gives_the_slots_back():
    # given
    launched, asked = [], []

    def visitors(calls):
        asked.append(calls)
        raise RiotBusy("Busy, try again in a minute.")

    evaluations = Evaluations(LocalTable(), _Riot(), launch=launched.append, visitor_budget=visitors, clock=lambda: START)

    # when
    with pytest.raises(RiotBusy):
        evaluations.refuse("visitor#one", "a#na1")
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "a#na1")

    # then
    assert asked == [1] and launched == ["puuid-a"] and evaluations.riot.calls == 1
    assert evaluations.table.get("visitor#one", f"evals#{evaluations.day()}")["used"] == 0
    assert evaluations.table.get(EVAL, f"visitors#{evaluations.day()}")["used"] == 0


def test_a_new_player_riot_refuses_answers_busy_and_gives_the_slots_back():
    # given
    launched = []
    evaluations = Evaluations(LocalTable(), _Riot(busy=True), launch=launched.append, clock=lambda: START)

    # when
    with pytest.raises(LinkError) as member:
        evaluations.refuse("user-1", "a#na1")
    with pytest.raises(RiotBusy) as visitor:
        evaluations.refuse("visitor#one", "a#na1")

    # then
    assert "Riot is busy" in str(member.value) and str(visitor.value) == "Busy, try again in a minute." and launched == []
    assert evaluations.table.get("user#user-1", f"evals#{evaluations.day()}")["used"] == 0
    assert evaluations.table.get("visitor#one", f"evals#{evaluations.day()}")["used"] == 0


def test_a_dead_riot_key_answers_plainly_gives_the_slots_back_and_queues_nothing():
    # given
    launched = []
    evaluations = Evaluations(LocalTable(), _Riot(down=True), launch=launched.append, clock=lambda: START)

    # when
    with pytest.raises(LinkError) as member:
        evaluations.refuse("user-1", "a#na1")
    with pytest.raises(LinkError) as visitor:
        evaluations.refuse("visitor#one", "a#na1")
    with pytest.raises(LinkError) as refreshed:
        evaluations.refresh("user-1", "a#na1")

    # then
    assert str(member.value) == str(visitor.value) == str(refreshed.value) == RIOT_DOWN == "Riot isn't answering us right now. Try again later."
    assert launched == [] and evaluations.status("puuid-a") is None
    assert evaluations.table.get("user#user-1", f"evals#{evaluations.day()}")["used"] == 0 and evaluations.table.get(EVAL, f"day#{evaluations.day()}")["used"] == 0
    assert evaluations.table.get("visitor#one", f"evals#{evaluations.day()}")["used"] == 0 and evaluations.table.get(EVAL, f"visitors#{evaluations.day()}")["used"] == 0


def test_only_a_player_failure_holds_the_player_for_a_day_while_ours_can_be_tried_again_at_once():
    # given
    launched = []
    now = {"t": START}
    evaluations = _evaluations(launched, clock=lambda: now["t"], daily=10, per_user=10)
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "a#na1")
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "b#na1")
    now["t"] += 600
    fail_evaluation(evaluations.table, "puuid-a", "Riot returned no ranked games for this player.", PLAYER, clock=lambda: now["t"])
    fail_evaluation(evaluations.table, "puuid-b", SYSTEM_PROBLEM, SYSTEM, clock=lambda: now["t"])

    # when
    with pytest.raises(LinkError) as held:
        evaluations.refuse("user-1", "a#na1")
    with pytest.raises(LinkError) as again:
        evaluations.refuse("user-1", "b#na1")
    now["t"] += 86400
    with pytest.raises(LinkError) as tomorrow:
        evaluations.refuse("user-1", "a#na1")

    # then
    assert "couldn't pull a#NA1's games. Try again tomorrow." in str(held.value) and "pulling their last" in str(again.value)
    assert shape(evaluations.table.get(EVAL, "puuid-b"), "b#NA1")["status"] == REQUESTED and "pulling their last" in str(tomorrow.value)
    assert launched == ["puuid-a", "puuid-b", "puuid-b", "puuid-a"]


def test_a_request_is_still_pulling_while_its_heartbeat_is_fresh_however_old_it_started():
    # given
    now = {"t": START}
    evaluations = _evaluations(clock=lambda: now["t"], daily=10, per_user=10)
    with pytest.raises(LinkError):
        evaluations.refuse("user-1", "a#na1")
    now["t"] += 3600
    Progress(evaluations.table, "puuid-a", clock=lambda: now["t"] - 60).step("embedding")

    # when
    with pytest.raises(LinkError) as busy:
        evaluations.refuse("user-1", "a#na1")

    # then
    assert "still pulling a#NA1's games" in str(busy.value) and evaluations.status("puuid-a")["started"] == int(START)


def test_our_failure_gives_the_requester_their_slot_back_exactly_once_on_the_pool_they_drew_from():
    # given
    evaluations = _evaluations(daily=10, per_user=10)
    evaluations.per_visitor = 10
    for user, name in (("user-1", "a#na1"), ("visitor#one", "b#na1"), ("user-1", "c#na1")):
        with pytest.raises(LinkError):
            evaluations.refuse(user, name)
    day, table = evaluations.day(), evaluations.table

    # when
    for _ in range(2):
        fail_evaluation(table, "puuid-a", SYSTEM_PROBLEM, SYSTEM, clock=lambda: START + 60)
        fail_evaluation(table, "puuid-b", SYSTEM_PROBLEM, SYSTEM, clock=lambda: START + 60)
    fail_evaluation(table, "puuid-c", "Riot returned no ranked games for this player.", PLAYER, clock=lambda: START + 60)

    # then
    assert table.get("user#user-1", f"evals#{day}")["used"] == 1 and table.get(EVAL, f"day#{day}")["used"] == 1
    assert table.get("visitor#one", f"evals#{day}")["used"] == 0 and table.get(EVAL, f"visitors#{day}")["used"] == 0
    assert table.get(EVAL, "puuid-a")["error"] == "We hit a problem pulling a#NA1's games. Try again." and "refunded" not in table.get(EVAL, "puuid-c")


def test_progress_is_stored_as_counts_and_the_percent_is_worked_out_when_read():
    # given
    table = LocalTable()
    table.put(EVAL, "me", {"status": "requested", "riot_id": "Me#NA1", "started": int(START)})
    progress = Progress(table, "me", clock=lambda: START)

    # when
    progress.step("fetching", 25, 50)
    middle = table.get(EVAL, "me")
    progress.step("embedding")
    late = shape(table.get(EVAL, "me"), "Me#NA1")
    progress.write(status=READY, step="ready")
    done = shape(table.get(EVAL, "me"), "Me#NA1")

    # then
    assert "percent" not in middle and shape(middle, "Me#NA1")["percent"] == 25 and middle["riot_id"] == "Me#NA1"
    assert late["step"] == "embedding" and late["percent"] == 92 and done["percent"] == 100
    assert table.get(EVAL, "me")["updated"] == int(START)
