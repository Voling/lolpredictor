import pandas as pd
import pytest

from synergy.ml.score import SynergyService
from synergy.ml.serving import NAMES_TABLE


@pytest.fixture
def service(serving_settings):
    service = SynergyService(serving_settings)
    service.model = object()
    service.profiles = pd.DataFrame(
        {
            "puuid": ["a", "b", "c"],
            "game_name": ["Alpha", "Beta", "Gamma"],
            "tag_line": ["NA1", "NA1", "NA1"],
            "games": [6, 7, 2],
            "winrate": [0.5, 0.5714, 0.5],
            "main_position": ["TOP", "JUNGLE", "TOP"],
        }
    ).set_index("puuid", drop=False)
    service.history = pd.DataFrame({"pair_key": ["a|b"], "games": [4], "wins": [3]}).set_index("pair_key", drop=False)
    return service


def test_a_pair_carries_its_position_score_on_top_and_keeps_the_reading_stats_below(service):
    # given
    left, right = "Alpha#NA1", "Beta#NA1"

    # when
    result = service.pair_score(left, right, "top", "jungle")

    # then
    interaction = result["interaction"]
    assert result["score"] == interaction["score"] == 64.0
    assert result["projected_gold_at_15"] == interaction["projected_gold_at_15"] == 5.0
    assert {"synergy", "percentile", "drivers", "reading", "edge", "left_evidence"} <= set(interaction)
    assert not {"synergy", "drivers"} & set(result)
    assert interaction["positions"] == {"left": "TOP", "right": "JUNGLE"}
    assert result["reliable"] is False and result["note"] == interaction["note"]
    assert result["games_together"] == 4 and result["winrate_together"] == 0.75
    assert result["hinge"] == {"left_reacts_to_right": None, "right_reacts_to_left": None}


def test_two_players_sharing_a_main_position_are_refused_without_positions(service):
    # given
    left, right = "Alpha#NA1", "Gamma#NA1"

    # when
    with pytest.raises(ValueError) as refused:
        service.pair_score(left, right)

    # then
    assert "both given TOP" in str(refused.value)


def test_a_player_with_no_games_in_the_asked_position_is_refused_by_name(service):
    # given
    left, right = "Beta#NA1", "Gamma#NA1"

    # when
    with pytest.raises(ValueError) as refused:
        service.pair_score(left, right, "jungle", "top")

    # then
    assert "Gamma#NA1 has no games as TOP" in str(refused.value)


def test_a_team_without_positions_reads_only_pairs_it_can_place(service):
    # given
    players = ["Alpha#NA1", "Beta#NA1", "Gamma#NA1"]

    # when
    report = service.team_report(players)

    # then
    by_pair = {(entry["a_riot_id"], entry["b_riot_id"]): entry for entry in report["pairs"]}
    assert by_pair[("Alpha#NA1", "Beta#NA1")]["percentile"] is not None
    assert by_pair[("Alpha#NA1", "Beta#NA1")]["score"] == 64.0
    assert by_pair[("Alpha#NA1", "Gamma#NA1")]["percentile"] is None
    assert by_pair[("Alpha#NA1", "Gamma#NA1")]["score"] is None
    assert report["team_score"] is None and report["reliable"] is False


def test_an_old_riot_id_resolves_through_the_exported_name_table(service):
    # given
    pd.DataFrame(
        {"puuid": ["a"], "game_name": ["Formerly"], "tag_line": ["NA1"], "current": [False]}
    ).to_parquet(service.settings.processed_dir / NAMES_TABLE, index=False)

    # when
    profile = service.resolve("Formerly#NA1")

    # then
    assert profile["puuid"] == "a"


def test_friends_are_ranked_by_the_projected_edge_and_a_refused_pair_keeps_its_reason(service):
    # given
    me, friends = "Alpha#NA1", ["Beta#NA1", "Gamma#NA1:top", "Nobody#NA1", " "]

    # when
    found = service.friends(me, friends)

    # then
    assert found["me"]["riot_id"] == "Alpha#NA1" and found["me"]["position"] == "TOP"
    assert found["me"]["reading"]["gold"] == 100.0
    assert [row["riot_id"] for row in found["friends"]] == ["Beta#NA1", "Gamma#NA1", "Nobody#NA1"]
    beta = found["friends"][0]
    assert beta["position"] == "JUNGLE" and beta["reading"]["gold"] == 50.0 and beta["total"] == 155.0
    assert beta["fit"]["gold"] == 5.0 and beta["games_together"] == 4 and beta["thin"] is False
    assert "both given TOP" in found["friends"][1]["note"]
    assert found["friends"][2]["note"] == "not in the corpus"
