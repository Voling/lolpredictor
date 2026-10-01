import json

import numpy as np

from synergy.api.accounts import LinkError
from synergy.api.customs import IDS, MATCHES, SharedCustoms, seats_at, together
from synergy.ml.serving import DUO_RECORDS, DUO_REPORT, custom_residuals, duo_between, record_between

START = 1_790_000_000
ME, FRIEND = "me", "friend"


class _Table:
    def __init__(self):
        self.items = {}

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None

    def put(self, pk, sk, values):
        self.items[(pk, sk)] = dict(values)


def _match(match_id, kind="CUSTOM_GAME", players=10, same_team=True, positions=None):
    positions = positions or ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    names = [ME, FRIEND, "c", "d", "e", "f", "g", "h", "i", "j"][:players]
    teams = [100 if index < 5 else 200 for index in range(players)]
    if not same_team:
        teams[1] = 200
        teams[5] = 100
    participants = [
        {"participantId": index + 1, "puuid": name, "teamId": teams[index], "teamPosition": positions[index % 5]}
        for index, name in enumerate(names)
    ]
    return {"metadata": {"matchId": match_id}, "info": {"gameType": kind, "participants": participants}}


def _timeline(frames=25, gold=None):
    gold = gold or {str(pid): 6000.0 + 100.0 * pid for pid in range(1, 11)}
    return {"info": {"frames": [{"participantFrames": {pid: {"totalGold": value} for pid, value in gold.items()}} for _ in range(frames)]}}


class _Riot:
    def __init__(self, ids, matches, timelines):
        self.ids, self.matches, self.timelines, self.calls = ids, matches, timelines, []

    def match_ids(self, puuid, kind, count=100):
        self.calls.append(("ids", puuid, kind))
        return list(self.ids.get(puuid, []))

    def match(self, match_id):
        self.calls.append(("match", match_id))
        if match_id not in self.matches:
            raise LinkError("Riot doesn't know that Riot ID. Check the name and tag.")
        return self.matches[match_id]

    def timeline(self, match_id):
        self.calls.append(("timeline", match_id))
        return self.timelines[match_id]


def _customs(riot, table=None, budget=lambda calls: True, clock=lambda: START):
    return SharedCustoms(riot, table or _Table(), budget, 20, clock)


def test_shared_customs_on_the_same_team_come_back_with_gold_at_the_minute_and_enemy_counterparts():
    # given
    riot = _Riot(
        {ME: ["NA1_3", "NA1_2", "NA1_1"], FRIEND: ["NA1_2", "NA1_1", "NA1_0"]},
        {"NA1_2": _match("NA1_2"), "NA1_1": _match("NA1_1", same_team=False)},
        {"NA1_2": _timeline(), "NA1_1": _timeline()},
    )

    # when
    games, complete = _customs(riot).between(ME, FRIEND)

    # then
    assert complete and [game["match_id"] for game in games] == ["NA1_2"]
    assert games[0]["seats"] == [
        {"puuid": ME, "position": "TOP", "gold": 6100.0, "enemy": "f", "enemy_gold": 6600.0},
        {"puuid": FRIEND, "position": "JUNGLE", "gold": 6200.0, "enemy": "g", "enemy_gold": 6700.0},
    ]
    assert ("match", "NA1_3") not in riot.calls and ("match", "NA1_0") not in riot.calls


def test_matches_and_id_lists_are_cached_so_riot_is_asked_once():
    # given
    table = _Table()
    riot = _Riot({ME: ["NA1_2", "NA1_9"], FRIEND: ["NA1_2", "NA1_9"]}, {"NA1_2": _match("NA1_2"), "NA1_9": _match("NA1_9", kind="MATCHED_GAME")}, {"NA1_2": _timeline()})
    customs = _customs(riot, table)

    # when
    first, _ = customs.between(ME, FRIEND)
    second, _ = customs.between(ME, FRIEND)

    # then
    assert first == second and len(first) == 1
    assert riot.calls.count(("match", "NA1_2")) == 1 and riot.calls.count(("match", "NA1_9")) == 1 and ("timeline", "NA1_9") not in riot.calls
    assert json.loads(table.get(MATCHES, "NA1_9")["seats"]) == [] and json.loads(table.get(IDS, ME)["ids"]) == ["NA1_2", "NA1_9"]


def test_an_id_list_is_refreshed_after_six_hours():
    # given
    now = {"t": START}
    riot = _Riot({ME: ["NA1_2"], FRIEND: ["NA1_2"]}, {"NA1_2": _match("NA1_2")}, {"NA1_2": _timeline()})
    customs = _customs(riot, clock=lambda: now["t"])
    customs.between(ME, FRIEND)

    # when
    now["t"] += 7 * 3600
    customs.between(ME, FRIEND)

    # then
    assert riot.calls.count(("ids", ME, "tourney")) == 2 and riot.calls.count(("match", "NA1_2")) == 1


def test_a_refused_riot_budget_returns_what_was_found_and_says_it_is_incomplete():
    # given
    allowance = {"left": 3}

    def budget(calls):
        allowance["left"] -= calls
        return allowance["left"] >= 0

    riot = _Riot({ME: ["NA1_2", "NA1_1"], FRIEND: ["NA1_2", "NA1_1"]}, {"NA1_2": _match("NA1_2"), "NA1_1": _match("NA1_1")}, {"NA1_2": _timeline(), "NA1_1": _timeline()})

    # when
    games, complete = _customs(riot, budget=budget).between(ME, FRIEND)

    # then
    assert games == [] and complete is False and ("timeline", "NA1_2") not in riot.calls


def test_short_games_odd_positions_and_other_queues_are_left_out():
    # given
    short = seats_at(_match("x"), _timeline(frames=15), 20)
    odd = seats_at(_match("x", positions=["TOP", "", "MIDDLE", "BOTTOM", "UTILITY"]), _timeline(), 20)
    fine = seats_at(_match("x"), _timeline(), 20)

    # when
    opposite = together(seats_at(_match("x", same_team=False), _timeline(), 20), ME, FRIEND)
    missing = together(fine, ME, "nobody")

    # then
    assert short == [] and odd == [] and len(fine) == 10
    assert opposite is None and missing is None


def test_custom_games_pool_into_the_record_with_the_corpus_shrinkage(serving_settings):
    # given
    settings = serving_settings
    import pandas as pd

    pd.DataFrame({"a": ["a"], "b": ["b"], "games": [4], "mean": [-1000.0], "record": [-200.0]}).to_parquet(settings.model_dir / DUO_RECORDS, index=False)
    (settings.model_dir / DUO_REPORT).write_text(json.dumps({"spread": 300.0, "noise": 3000.0}), encoding="utf-8")

    # when
    alone = record_between("a", "b", settings)
    with_customs = record_between("b", "a", settings, [2000.0, 2000.0, 2000.0, 2000.0])
    unknown = record_between("x", "y", settings, [500.0])

    # then
    assert alone == {"gold": -200.0, "games": 4, "customs": 0}
    assert with_customs["games"] == 8 and with_customs["customs"] == 4
    assert with_customs["gold"] == round(500.0 * 90000.0 / (90000.0 + 9e6 / 8), 1)
    assert unknown == {"gold": round(500.0 * 90000.0 / (90000.0 + 9e6), 1), "games": 1, "customs": 1}


def test_custom_residuals_subtract_known_readings_and_treat_strangers_as_average(serving_settings):
    # given
    settings = serving_settings
    game = {"seats": [
        {"puuid": "a", "position": "TOP", "gold": 7000.0, "enemy": "nobody", "enemy_gold": 6000.0},
        {"puuid": "b", "position": "JUNGLE", "gold": 6500.0, "enemy": "a", "enemy_gold": 6500.0},
    ]}
    expected = duo_between("a", "TOP", "b", "JUNGLE", settings)["edge"]

    # when
    residuals = custom_residuals([game], settings)

    # then
    own = expected["left"]["gold"] + expected["right"]["gold"]
    assert len(residuals) == 1 and np.isclose(residuals[0], 1000.0 - (own - _reading_of("a", "JUNGLE", settings)), atol=0.6)


def _reading_of(puuid, position, settings):
    from synergy.ml import serving

    loaded = serving._loaded(settings)
    return serving._expected(loaded, puuid, position, settings)


def test_a_pair_score_passes_shared_customs_through_to_the_record(serving_settings):
    # given
    settings = serving_settings
    import pandas as pd

    from synergy.ml.score import SynergyService

    pd.DataFrame({"a": ["a"], "b": ["b"], "games": [2], "mean": [0.0], "record": [0.0]}).to_parquet(settings.model_dir / DUO_RECORDS, index=False)
    (settings.model_dir / DUO_REPORT).write_text(json.dumps({"spread": 300.0, "noise": 3000.0}), encoding="utf-8")
    np.savez(settings.model_dir / "duo_scores.npz", quantiles=np.linspace(-400.0, 400.0, 1001), with_fit=np.array(False))
    service = SynergyService(settings)
    service.model, service.profiles = object(), pd.DataFrame({"puuid": ["a", "b"], "game_name": ["a", "b"], "tag_line": ["NA1", "NA1"], "games": [5, 7], "winrate": [0.5, 0.5], "main_position": ["TOP", "JUNGLE"]}).set_index("puuid", drop=False)
    asked = []

    def customs(left, right):
        asked.append((left, right))
        return [{"seats": [
            {"puuid": "a", "position": "TOP", "gold": 9000.0, "enemy": "x", "enemy_gold": 6000.0},
            {"puuid": "b", "position": "JUNGLE", "gold": 9000.0, "enemy": "y", "enemy_gold": 6000.0},
        ]}]

    # when
    found = service.pair_score("a#NA1", "b#NA1", details=False, customs=customs)

    # then
    assert asked == [("a", "b")] and found["interaction"]["edge"]["record"]["customs"] == 1
    assert found["interaction"]["edge"]["record"]["games"] == 3 and found["interaction"]["edge"]["record"]["gold"] > 0.0
