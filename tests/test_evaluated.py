import json

import numpy as np
import pandas as pd

from synergy.api.evaluate import EVAL, NAMES, READY
from synergy.ml import serving
from synergy.ml.evaluated import EvaluatedPlayers, pool_effect, profile_row, seats_of
from synergy.ml.serving import CHAMPION_EFFECTS, adopt_player, duo_between, position_profile

START = 1_790_000_000.0


class _Table:
    def __init__(self, items):
        self.items = items

    def get(self, pk, sk):
        found = self.items.get((pk, sk))
        return dict(found) if found else None


class _S3:
    def __init__(self, profile, vectors):
        self.profile, self.vectors, self.calls = profile, vectors, 0

    def get_object(self, Bucket, Key):
        self.calls += 1

        class _Body:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

        return {"Body": _Body(json.dumps(self.profile).encode())}

    def download_file(self, Bucket, Key, Filename):
        self.calls += 1
        np.savez(Filename, **self.vectors)


def _vectors(columns):
    return {
        "columns": np.array(columns),
        "position": np.array(["TOP", "JUNGLE"]),
        "seats": np.array([30, 6], dtype=np.int32),
        "evidence": np.array([0.7, 0.2], dtype=np.float32),
        "matrix": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }


def _profile():
    return {"puuid": "new", "game_name": "New", "tag_line": "NA1", "games": 36, "winrate": 0.5, "main_position": "TOP", "main_champion": "Garen", "tier": "GOLD", "division": "II", "positions": ["TOP", "JUNGLE"], "seats": [30, 6], "champions": {"TOP": {"Garen": 20, "Darius": 10}, "JUNGLE": {"Amumu": 6}}}


def test_a_ready_player_is_fetched_once_and_kept_for_ten_minutes(tmp_path):
    # given
    now = {"t": START}
    table = _Table({(NAMES, "new#na1"): {"puuid": "new"}, (EVAL, "new"): {"status": READY, "expires": int(START) + 1000}})
    client = _S3(_profile(), _vectors(["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]))
    players = EvaluatedPlayers(table, client, "bucket", tmp_path, clock=lambda: now["t"])

    # when
    first = players.find(" New#NA1 ")
    now["t"] += 100
    again = players.find("new#na1")
    now["t"] += 1000
    expired = players.find("new#na1")

    # then
    assert first is not None and first[0]["main_champion"] == "Garen" and list(first[1]["position"]) == ["TOP", "JUNGLE"]
    assert again == first and client.calls == 2 and expired is None


def test_an_evaluated_player_joins_the_served_pack_with_a_champion_baseline_and_scores_like_anyone(serving_settings, tmp_path):
    # given
    settings = serving_settings
    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    pd.DataFrame({"position": ["TOP", "TOP", "JUNGLE"], "champion": ["Garen", "Darius", "Amumu"], "games": [100, 100, 100], "effect": [60.0, -30.0, 10.0]}).to_parquet(settings.model_dir / CHAMPION_EFFECTS, index=False)
    vectors = {name: value for name, value in _vectors(columns).items()}
    vectors["columns"] = [str(name) for name in vectors["columns"]]
    vectors["position"] = [str(name) for name in vectors["position"]]
    duo_between("a", "TOP", "b", "JUNGLE", settings)

    # when
    adopt_player(settings, _profile(), vectors)
    found = duo_between("new", "TOP", "b", "JUNGLE", settings)

    # then
    assert position_profile("new", "TOP", settings) == {"games": 30, "evidence": 0.7}
    assert found is not None and found["left_games"] == 30 and found["edge"]["left"]["champion"] == 30.0
    assert serving.player_history("new", "JUNGLE", settings) == (0.0, 10.0, 6)
    assert np.isclose(pool_effect(settings, "TOP", {"Garen": 20, "Darius": 10}), 30.0)


def test_a_profile_row_and_seat_rows_keep_only_what_the_service_reads():
    # given
    profile = _profile()
    vectors = {**_vectors(["b", "a"]), "puuid": "new"}
    vectors["columns"], vectors["position"] = ["b", "a"], ["TOP", "JUNGLE"]

    # when
    row = profile_row(profile)
    seats = seats_of(vectors, ["a", "b"])

    # then
    assert row.name == "new" and row["evaluated"] is True and "champions" not in row.index and row["games"] == 36
    assert np.allclose(seats[("new", "TOP")][0], [0.0, 1.0]) and seats[("new", "JUNGLE")][1] == 6 and np.isclose(seats[("new", "JUNGLE")][2], 0.2)


def test_an_adopted_player_fills_the_profile_columns_the_summary_reads(serving_settings):
    # given
    from synergy.ml.score import SynergyService

    settings = serving_settings
    service = SynergyService(settings)
    service.model = object()
    service.profiles = pd.DataFrame({"puuid": ["a"], "game_name": ["a"], "tag_line": ["NA1"], "games": [5], "winrate": [0.5], "main_position": ["TOP"], "chances_initiate": [3], "style_dive_pct": [70.0], "champion_pool": [2]}).set_index("puuid", drop=False)
    vectors = {**_vectors(["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]), "puuid": "new"}
    vectors["columns"], vectors["position"] = [str(c) for c in vectors["columns"]], ["TOP", "JUNGLE"]

    # when
    row = service._adopt(_profile(), vectors)
    summary = service.player_summary(service.profiles.loc["new"])

    # then
    assert row["games"] == 36 and row["chances_initiate"] == 0.0 and row["style_dive_pct"] == 50.0 and row["champion_pool"] == 0.0
    assert summary["riot_id"] == "New#NA1" and summary["tendencies"]["initiate"]["chances"] == 0 and service.profiles.loc["new", "puuid"] == "new"


def test_adopting_a_second_player_keeps_the_profile_columns_numeric_so_its_summary_still_reads(serving_settings):
    # given
    from synergy.ml.score import SynergyService

    service = SynergyService(serving_settings)
    service.model = object()
    service.profiles = pd.DataFrame({"puuid": ["a"], "game_name": ["a"], "tag_line": ["NA1"], "games": [5], "winrate": [0.5], "main_position": ["TOP"], "chances_initiate": [3], "style_dive_pct": [70.0], "champion_pool": [2]}).set_index("puuid", drop=False)
    vectors = {**_vectors(["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]), "puuid": "new"}
    vectors["columns"], vectors["position"] = [str(c) for c in vectors["columns"]], ["TOP", "JUNGLE"]
    second = {**_profile(), "puuid": "other", "game_name": "Other"}

    # when
    service._adopt(_profile(), vectors)
    dtypes = service.profiles.dtypes
    service._adopt(second, {**vectors, "puuid": "other"})
    summary = service.player_summary(service.profiles.loc["other"])

    # then
    assert dtypes["games"].kind == "i" and dtypes["winrate"].kind == "f" and dtypes["chances_initiate"].kind == "i"
    assert summary["riot_id"] == "Other#NA1" and summary["games"] == 36 and summary["tendencies"]["initiate"]["chances"] == 0
    assert service.profiles.loc["other", "style_dive_pct"] == 50.0 and service.profiles.loc["other", "champion_pool"] == 0


def test_a_newer_evaluation_is_fetched_again_and_replaces_what_was_adopted(serving_settings, tmp_path):
    # given
    from synergy.ml.score import SynergyService

    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    table = _Table({(EVAL, "a"): {"status": READY, "expires": int(START) + 1000, "finished": 100}})
    client = _S3({**_profile(), "puuid": "a", "game_name": "Alpha", "games": 40, "latest_game": 1_789_000_000}, _vectors(columns))
    service = SynergyService(serving_settings)
    service.model = object()
    service.evaluated = EvaluatedPlayers(table, client, "bucket", tmp_path, clock=lambda: START)
    service.profiles = pd.DataFrame({"puuid": ["a"], "game_name": ["Alpha"], "tag_line": ["NA1"], "games": [5], "winrate": [0.5], "main_position": ["TOP"], "style_dive_pct": [70.0]}).set_index("puuid", drop=False)

    # when
    first = service.resolve("Alpha#NA1")
    calls_first = client.calls
    again = service.resolve("Alpha#NA1")
    calls_again = client.calls
    table.items[(EVAL, "a")]["finished"] = 200
    client.profile["games"] = 48
    newer = service.resolve("a")
    summary = service.player_summary(newer)

    # then
    assert first["games"] == 40 and first["style_dive_pct"] == 70.0 and first["evaluated"] is True and calls_first == 2
    assert again["games"] == 40 and calls_again == 2
    assert newer["games"] == 48 and client.calls == 4 and service.adopted["a"] == 200 and len(service.profiles) == 1
    assert summary["latest_game"] == 1_789_000_000 and summary["refresh_after"] == 1_789_000_000 + 14 * 86400


def test_an_adopted_seat_takes_precedence_over_the_packed_one(serving_settings):
    # given
    settings = serving_settings
    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    duo_between("a", "TOP", "b", "JUNGLE", settings)
    before = position_profile("a", "TOP", settings)
    vectors = {**_vectors(columns), "columns": columns, "position": ["TOP", "JUNGLE"]}

    # when
    adopt_player(settings, {**_profile(), "puuid": "a"}, vectors)

    # then
    assert before["games"] > 0 and position_profile("a", "TOP", settings) == {"games": 30, "evidence": 0.7}


def test_champions_follow_the_seat_position_and_an_evaluated_pool_wins_over_the_corpus(serving_settings):
    # given
    from synergy.features.pools import PLAYER_CHAMPIONS
    from synergy.ml.serving import champions_of

    settings = serving_settings
    pd.DataFrame({"puuid": ["a", "a"], "position": ["TOP", "JUNGLE"], "champions": ["Pantheon,Zaahen,KSante", "Graves,Udyr"]}).to_parquet(settings.processed_dir / PLAYER_CHAMPIONS, index=False)
    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    duo_between("a", "TOP", "b", "JUNGLE", settings)
    vectors = {**_vectors(columns), "columns": columns, "position": ["TOP", "JUNGLE"]}

    # when
    before = (champions_of("a", "TOP", settings), champions_of("a", "JUNGLE", settings), champions_of("a", "BOTTOM", settings))
    adopt_player(settings, {**_profile(), "puuid": "a"}, vectors)
    after = champions_of("a", "TOP", settings)

    # then
    assert before == (["Pantheon", "Zaahen", "KSante"], ["Graves", "Udyr"], [])
    assert after == ["Garen", "Darius"] and champions_of("a", "JUNGLE", settings) == ["Amumu"]


def test_vectors_from_before_new_columns_still_load_with_the_new_columns_typical():
    # given
    import pytest

    vectors = {"puuid": "new", "columns": [f"c{index}" for index in range(10)], "position": ["TOP"], "seats": np.array([12]), "evidence": np.array([0.5]), "matrix": np.arange(10, dtype=np.float32)[None, :]}

    # when
    grown = seats_of(vectors, [*vectors["columns"], "c_new"])

    # then
    row, seats, evidence = grown[("new", "TOP")]
    assert row.tolist() == [*range(10), 0.0] and seats == 12 and evidence == 0.5
    with pytest.raises(ValueError):
        seats_of(vectors, [*vectors["columns"][:5], *(f"other{index}" for index in range(6))])

