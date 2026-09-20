import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synergy.config import Settings
from synergy.ml.serving import (
    NAMES_TABLE,
    NoGamesInPosition,
    cached,
    known_names,
    lineup_between,
    pair_between,
    position_profile,
)


def test_pair_between_reads_each_player_in_the_given_position_and_ranks_within_the_position_pair(serving_settings):
    # given
    settings = serving_settings

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    unrelated = pair_between("a", "TOP", "c", "MIDDLE", settings)
    off_role = pair_between("a", "JUNGLE", "c", "MIDDLE", settings)

    # then
    assert np.isclose(alike["synergy"], 1.0, atol=1e-6) and alike["percentile"] > 50
    assert alike["drivers"][0]["family"] == "tend" and alike["drivers"][0]["words"] == "fight and lane tendencies"
    assert np.isclose(alike["drivers"][0]["contribution"], 1.0, atol=1e-6)
    assert alike["left_games"] == 5 and alike["right_games"] == 7
    assert alike["positions"] == {"left": "TOP", "right": "JUNGLE"}
    assert np.isclose(unrelated["synergy"], 0.0, atol=1e-6)
    assert np.isclose(off_role["synergy"], -1.0, atol=1e-6) and off_role["left_games"] == 1
    assert alike["reliable"] is False and "inspection only" in alike["note"]
    reading = alike["reading"]["left"]
    assert reading["distinctive"][0]["cell"] == "tend_dive_tmb_own" and "dives" in reading["distinctive"][0]["words"]
    assert reading["distinctive"][0]["percentile"] == 0.0


def test_the_order_of_the_two_players_never_changes_the_fit(serving_settings):
    # given
    settings = serving_settings

    # when
    forward = pair_between("a", "TOP", "b", "JUNGLE", settings)
    backward = pair_between("b", "JUNGLE", "a", "TOP", settings)

    # then
    assert np.isclose(forward["synergy"], backward["synergy"], atol=1e-9)
    assert forward["edge"]["left"] == backward["edge"]["right"]
    assert forward["edge"]["total"] == backward["edge"]["total"]


def test_a_pair_scores_fifty_at_the_average_and_reports_its_fit_in_gold_at_15(serving_settings):
    # given
    settings = serving_settings

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    unrelated = pair_between("a", "TOP", "c", "MIDDLE", settings)
    off_role = pair_between("a", "JUNGLE", "c", "MIDDLE", settings)

    # then
    assert unrelated["score"] == 50.0 and unrelated["projected_gold_at_15"] == 0.0
    assert alike["score"] == 64.0 and alike["projected_gold_at_15"] == 1.0
    assert off_role["score"] == 36.0 and off_role["projected_gold_at_15"] == -1.0


def test_the_edge_splits_into_each_seat_and_the_fit_and_sums_to_the_total(serving_settings):
    # given
    settings = serving_settings

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    edge = alike["edge"]

    # then
    assert edge["left"]["gold"] == 100.0 and edge["right"]["gold"] == 50.0 and edge["fit"]["gold"] == 1.0
    assert edge["total"] == 151.0
    assert edge["fit"]["error"] == 0.0 and edge["fit"]["known"] == 0.35
    assert edge["left"]["score"] > 50.0 and edge["left"]["percentile"] > 50.0
    assert edge["left"]["score"] > edge["right"]["score"]


def test_a_pair_keeps_values_far_below_a_millionth_instead_of_rounding_them_to_zero(serving_settings):
    # given
    path = serving_settings.model_dir / "pairnet.npz"
    with np.load(path, allow_pickle=True) as data:
        saved = {name: data[name] for name in data.files}
    np.savez(path, **{**saved, "scale": saved["scale"] * 1e-7})

    # when
    found = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)

    # then
    assert np.isclose(found["synergy"], 1e-7, rtol=1e-6, atol=0.0)
    assert found["drivers"][0]["contribution"] == 1e-7


def test_the_reliability_flag_follows_the_fitted_report_as_soon_as_it_is_written(serving_settings):
    # given
    before = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)["reliable"]

    # when
    (serving_settings.model_dir / "pairnet_report.json").write_text('{"informative": true}', encoding="utf-8")
    after = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)

    # then
    assert before is False
    assert after["reliable"] is True and after["note"] is None


def test_pair_between_refuses_a_shared_position_and_a_position_never_played(serving_settings, tmp_path):
    # given
    settings = serving_settings

    # when
    with pytest.raises(ValueError) as shared:
        pair_between("a", "TOP", "b", "TOP", settings)
    with pytest.raises(NoGamesInPosition) as never:
        pair_between("a", "TOP", "b", "UTILITY", settings)

    # then
    assert "two different positions" in str(shared.value)
    assert never.value.position == "UTILITY" and "no games as UTILITY" in str(never.value)
    assert position_profile("b", "UTILITY", settings) == {"games": 0, "evidence": 0.0}
    assert position_profile("b", "JUNGLE", settings) == {"games": 7, "evidence": 0.9}
    assert position_profile("b", "JUNGLE", Settings(data_dir=tmp_path / "empty")) is None
    assert pair_between("a", "TOP", "b", "JUNGLE", settings)["right_evidence"] == 0.9


def test_a_lineup_sums_its_ten_pairs_and_ranks_against_corpus_teams(serving_settings):
    # given
    five = {"TOP": "a", "JUNGLE": "b", "MIDDLE": "c", "BOTTOM": "d", "UTILITY": "e"}

    # when
    result = lineup_between(five, serving_settings)
    with pytest.raises(ValueError):
        lineup_between({**five, "UTILITY": "a"}, serving_settings)
    with pytest.raises(ValueError):
        lineup_between({key: value for key, value in five.items() if key != "UTILITY"}, serving_settings)

    # then
    assert len(result["pairs"]) == 10
    assert np.isclose(result["synergy"], sum(pair["synergy"] for pair in result["pairs"]), rtol=1e-12, atol=1e-15)
    assert 0.0 <= result["percentile"] <= 100.0
    assert 0.0 < result["score"] < 100.0 and result["projected_gold_at_15"] == round(result["synergy"], 2)
    assert all(0.0 < pair["score"] < 100.0 for pair in result["pairs"])
    assert result["pairs"][0]["synergy"] >= result["pairs"][-1]["synergy"]
    assert result["edge"]["seats"] == {"TOP": 100.0, "JUNGLE": 50.0, "MIDDLE": 40.0, "BOTTOM": 30.0, "UTILITY": 10.0}
    assert np.isclose(result["edge"]["total"], 230.0 + result["synergy"], atol=0.1)


def test_a_cached_file_is_read_once_and_again_only_after_it_changes(tmp_path):
    # given
    path = tmp_path / "value.json"
    path.write_text('{"n": 1}', encoding="utf-8")
    reads = []

    def load(target):
        reads.append(target)
        return json.loads(target.read_text(encoding="utf-8"))

    # when
    first = cached(path, "value", load)
    second = cached(path, "value", load)
    path.write_text('{"n": 22}', encoding="utf-8")
    third = cached(path, "value", load)
    path.unlink()
    gone = cached(path, "value", load)

    # then
    assert first == second == {"n": 1}
    assert third == {"n": 22}
    assert len(reads) == 2
    assert gone is None


def test_a_riot_id_resolves_to_its_current_holder_before_an_older_one(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    pd.DataFrame(
        {
            "puuid": ["old-holder", "new-holder", "renamed"],
            "game_name": ["Same", "Same", "Before"],
            "tag_line": ["NA1", "na1", "EUW"],
            "current": [False, True, False],
        }
    ).to_parquet(settings.processed_dir / NAMES_TABLE, index=False)

    # when
    names = known_names(settings)

    # then
    assert names[("same", "na1")] == "new-holder"
    assert names[("before", "euw")] == "renamed"


def test_serving_a_request_never_loads_the_training_libraries():
    # given
    probe = (
        "import sys; import synergy.api.server, synergy.ml.score, synergy.ml.serving; "
        "print(','.join(name for name in ('torch', 'jax', 'numpyro') if name in sys.modules))"
    )

    # when
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    # then
    assert result.stdout.strip() == ""
