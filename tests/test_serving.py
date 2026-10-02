import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synergy.config import Settings
from synergy.ml import serving
from synergy.ml.serving import (
    NAMES_TABLE,
    DUO_RECORDS,
    DUO_SCORES,
    PLAYER_HISTORY,
    NoGamesInPosition,
    cached,
    duo_between,
    known_names,
    GRID,
    VECTORS_FILE,
    _grid,
    lineup_between,
    percentile_among,
    warm,
    write_vectors,
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
    assert alike["synergy"] > 0 and alike["percentile"] > 50
    assert alike["drivers"][0]["left"] == "tend_dive_tmb_own" and alike["drivers"][0]["right"] == "tend_dive_tmb_own"
    assert alike["left_games"] == 5 and alike["right_games"] == 7
    assert alike["positions"] == {"left": "TOP", "right": "JUNGLE"}
    assert unrelated["synergy"] == 0.0
    assert off_role["synergy"] < 0 and off_role["left_games"] == 1
    assert alike["reliable"] is False and "inspection only" in alike["note"]
    reading = alike["reading"]["left"]
    assert reading["distinctive"][0]["cell"] == "tend_dive_tmb_own" and "dives" in reading["distinctive"][0]["words"]
    assert reading["distinctive"][0]["percentile"] == 0.0
    assert np.isclose(sum(item["contribution"] for item in reading["situations"]), alike["synergy"], atol=1e-5)
    assert reading["situations"][0]["words"] == "diving"


def test_the_order_of_the_two_players_never_changes_the_fit(serving_settings):
    # given
    settings = serving_settings

    # when
    forward = pair_between("a", "TOP", "b", "JUNGLE", settings)
    backward = pair_between("b", "JUNGLE", "a", "TOP", settings)

    # then
    assert np.isclose(forward["synergy"], backward["synergy"], atol=1e-12)
    assert forward["edge"]["left"] == backward["edge"]["right"]
    assert forward["edge"]["total"] == backward["edge"]["total"]


def test_a_pair_scores_fifty_at_the_average_and_projects_its_fit_into_gold_at_15(serving_settings):
    # given
    settings = serving_settings

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    unrelated = pair_between("a", "TOP", "c", "MIDDLE", settings)
    off_role = pair_between("a", "JUNGLE", "c", "MIDDLE", settings)

    # then
    assert unrelated["score"] == 50.0 and unrelated["projected_gold"] == 0.0
    assert alike["score"] == 64.0 and alike["projected_gold"] == 5.0
    assert off_role["score"] == 36.0 and off_role["projected_gold"] == -5.0


def test_the_edge_splits_into_each_seat_and_the_fit_and_sums_to_the_total(serving_settings):
    # given
    settings = serving_settings

    # when
    alike = pair_between("a", "TOP", "b", "JUNGLE", settings)
    edge = alike["edge"]

    # then
    assert edge["left"]["gold"] == 100.0 and edge["right"]["gold"] == 50.0 and edge["fit"]["gold"] == 5.0
    assert edge["total"] == 155.0
    assert edge["left"]["score"] > 50.0 and edge["left"]["percentile"] > 50.0
    assert edge["left"]["score"] > edge["right"]["score"]


def test_the_seat_reading_applies_its_calibration_scale(serving_settings):
    # given
    path = serving_settings.model_dir / "seat_weights.npz"
    with np.load(path, allow_pickle=True) as data:
        saved = {name: data[name] for name in data.files}
    np.savez(path, **{**saved, "scale": np.array([1.2, 1.0, 1.0, 1.0, 1.0])})

    # when
    found = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)

    # then
    assert found["edge"]["left"]["gold"] == 120.0 and found["edge"]["right"]["gold"] == 50.0


def test_a_pair_keeps_values_far_below_a_millionth_instead_of_rounding_them_to_zero(serving_settings):
    # given
    path = serving_settings.model_dir / "interaction_matrix.npz"
    with np.load(path, allow_pickle=True) as data:
        saved = {name: data[name] for name in data.files}
    np.savez(path, **{**saved, "matrix": saved["matrix"] * 1e-7})

    # when
    found = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)

    # then
    assert np.isclose(found["synergy"], 5e-9, rtol=1e-12, atol=0.0)
    assert found["drivers"][0]["contribution"] == 5e-9
    assert found["reading"]["left"]["situations"][0]["contribution"] == 5e-9


def test_the_reliability_flag_follows_the_fitted_report_as_soon_as_it_is_written(serving_settings):
    # given
    before = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)["reliable"]

    # when
    (serving_settings.model_dir / "interaction_report.json").write_text('{"informative": true}', encoding="utf-8")
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
    assert never.value.position == "UTILITY" and "no games as support" in str(never.value)
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
    assert 0.0 < result["score"] < 100.0 and result["projected_gold"] == round(result["synergy"] * 100.0, 2)
    assert all(0.0 < pair["score"] < 100.0 for pair in result["pairs"])
    assert result["pairs"][0]["synergy"] >= result["pairs"][-1]["synergy"]
    assert result["edge"]["seats"] == {"TOP": 100.0, "JUNGLE": 50.0, "MIDDLE": 40.0, "BOTTOM": 30.0, "UTILITY": 10.0}
    assert np.isclose(result["edge"]["total"], 230.0 + result["synergy"] * 100.0, atol=0.1)


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


def _duo_artifacts(settings, records: list[tuple[str, str, int, float]], combos: list[str]) -> None:
    pd.DataFrame(
        {
            "a": [row[0] for row in records],
            "b": [row[1] for row in records],
            "games": [row[2] for row in records],
            "mean": [row[3] * 2.0 for row in records],
            "record": [row[3] for row in records],
        }
    ).to_parquet(settings.model_dir / DUO_RECORDS, index=False)
    grid = np.linspace(-400.0, 400.0, 1001)
    np.savez(settings.model_dir / DUO_SCORES, quantiles=grid, combos=np.array(combos), combo_quantiles=np.stack([grid] * len(combos)))


def test_the_duo_score_adds_the_record_together_to_both_readings_and_the_fit(serving_settings):
    # given
    _duo_artifacts(serving_settings, [("a", "b", 12, 40.0)], ["JUNGLE+TOP"])

    # when
    forward = duo_between("a", "TOP", "b", "JUNGLE", serving_settings)
    backward = duo_between("b", "JUNGLE", "a", "TOP", serving_settings)

    # then
    assert forward["projected_gold"] == backward["projected_gold"] == 195.0
    assert forward["edge"]["record"] == {"gold": 40.0, "games": 12, "customs": 0}
    assert forward["edge"]["total"] == 195.0 and forward["edge"]["fit"]["gold"] == 5.0
    assert forward["percentile"] == 74.4 and forward["score"] == backward["score"] == 57.0


def test_a_duo_never_seen_together_scores_on_its_readings_and_fit_alone(serving_settings):
    # given
    _duo_artifacts(serving_settings, [("a", "b", 12, 40.0)], ["JUNGLE+TOP", "MIDDLE+TOP"])

    # when
    found = duo_between("a", "TOP", "c", "MIDDLE", serving_settings)

    # then
    assert found["edge"]["record"] == {"gold": 0.0, "games": 0, "customs": 0}
    assert found["projected_gold"] == 140.0


def test_without_duo_scores_a_pair_keeps_the_fit_on_top(serving_settings):
    # given
    settings = serving_settings

    # when
    duo = duo_between("a", "TOP", "b", "JUNGLE", settings)
    pair = pair_between("a", "TOP", "b", "JUNGLE", settings)

    # then
    assert duo == pair


def test_a_reading_adds_the_players_form_and_champion_baseline_and_ranks_among_such_readings(serving_settings):
    # given
    pd.DataFrame({"puuid": ["a"], "position": ["TOP"], "games": [30], "form": [40.0], "champion": [25.0]}).to_parquet(
        serving_settings.model_dir / PLAYER_HISTORY, index=False
    )
    grid = np.linspace(-400.0, 400.0, 1001)
    np.savez(
        serving_settings.model_dir / DUO_SCORES,
        quantiles=grid,
        combos=np.array(["JUNGLE+TOP"]),
        combo_quantiles=np.stack([grid]),
        seat_positions=np.array(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]),
        seat_quantiles=np.stack([np.linspace(-200.0, 200.0, 1001)] * 5),
    )

    # when
    found = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)
    left, right = found["edge"]["left"], found["edge"]["right"]

    # then
    assert left["gold"] == 165.0 and left["style"] == 100.0 and left["form"] == 40.0 and left["champion"] == 25.0
    assert right["gold"] == 50.0 and right["form"] == 0.0 and right["champion"] == 0.0
    assert found["edge"]["total"] == 220.0
    assert left["percentile"] == 91.3 and right["percentile"] == 62.5


def test_a_reading_is_shrunk_by_its_held_out_calibration_and_its_parts_still_add_up(serving_settings):
    # given
    pd.DataFrame({"puuid": ["a"], "position": ["TOP"], "games": [30], "form": [40.0], "champion": [25.0]}).to_parquet(
        serving_settings.model_dir / PLAYER_HISTORY, index=False
    )
    grid = np.linspace(-400.0, 400.0, 1001)
    np.savez(
        serving_settings.model_dir / DUO_SCORES,
        quantiles=grid,
        combos=np.array(["JUNGLE+TOP"]),
        combo_quantiles=np.stack([grid]),
        seat_positions=np.array(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]),
        seat_quantiles=np.stack([np.linspace(-200.0, 200.0, 1001)] * 5),
        reading_scale=np.array([0.5, 1.0, 1.0, 1.0, 1.0]),
    )

    # when
    left = pair_between("a", "TOP", "b", "JUNGLE", serving_settings)["edge"]["left"]

    # then
    assert left["gold"] == 82.5
    assert (left["style"], left["form"], left["champion"]) == (50.0, 20.0, 12.5)


def test_a_style_fit_that_failed_its_shuffles_is_left_out_of_the_duo_score(serving_settings):
    # given
    grid = np.linspace(-400.0, 400.0, 1001)
    pd.DataFrame({"a": ["a"], "b": ["b"], "games": [12], "mean": [80.0], "record": [40.0]}).to_parquet(
        serving_settings.model_dir / DUO_RECORDS, index=False
    )
    np.savez(serving_settings.model_dir / DUO_SCORES, quantiles=grid, combos=np.array(["JUNGLE+TOP"]), combo_quantiles=np.stack([grid]), with_fit=False)

    # when
    found = duo_between("a", "TOP", "b", "JUNGLE", serving_settings)

    # then
    assert found["projected_gold"] == found["edge"]["total"] == 100.0 + 50.0 + 40.0
    assert found["edge"]["fit"]["gold"] == 5.0 and found["edge"]["with_fit"] is False
    assert found["reliable"] is True and found["note"] is None


def test_the_packed_vectors_serve_the_same_pair_as_the_style_table(serving_settings):
    # given
    settings = serving_settings
    from_table = pair_between("a", "TOP", "b", "JUNGLE", settings)

    # when
    written = write_vectors(settings.model_dir, settings.processed_dir)
    (settings.processed_dir / "player_styles.parquet").unlink()
    from_pack = pair_between("a", "TOP", "b", "JUNGLE", settings)

    # then
    assert written == settings.model_dir / VECTORS_FILE and from_pack == from_table
    assert position_profile("b", "JUNGLE", settings) == {"games": 7, "evidence": 0.9}


def test_the_pack_keeps_only_profiled_players_but_ranks_them_among_everyone(serving_settings):
    # given
    settings = serving_settings
    pd.DataFrame({"puuid": ["b", "c"]}).to_parquet(settings.processed_dir / "player_profiles.parquet", index=False)

    # when
    write_vectors(settings.model_dir, settings.processed_dir)
    (settings.processed_dir / "player_styles.parquet").unlink()
    found = pair_between("b", "JUNGLE", "c", "MIDDLE", settings)

    # then
    assert position_profile("a", "TOP", settings) == {"games": 0, "evidence": 0.0}
    assert found["reading"]["left"]["distinctive"][0]["percentile"] == 50.0


def test_a_grid_percentile_matches_counting_the_peers_below_to_a_tenth():
    # given
    rng = np.random.default_rng(3)
    values = rng.normal(size=(20_000, 1)).astype(np.float32)
    grid = _grid(values)[0]
    probes = rng.normal(size=50)

    # when
    exact = [round(float((values[:, 0] < probe).mean() * 100.0), 1) for probe in probes]
    approximate = [percentile_among(grid, probe) for probe in probes]

    # then
    assert len(grid) == GRID and max(abs(a - e) for a, e in zip(approximate, exact)) < 0.11


def test_warming_up_loads_every_served_table_once(serving_settings, monkeypatch):
    # given
    settings = serving_settings
    _duo_artifacts(settings, [("a", "b", 10, 50.0)], ["JUNGLE+TOP"])
    loads = []
    original = serving.cached
    monkeypatch.setattr(serving, "cached", lambda path, kind, load: original(path, kind, lambda found: loads.append(kind) or load(found)))

    # when
    warm(settings)
    again = len(loads)
    warm(settings)

    # then
    assert "vectors:2:" in "".join(loads) and "npz" in loads and len(loads) == again


def test_a_cell_on_which_players_do_not_differ_is_never_a_standout():
    # given
    from synergy.ml.serving import _reading

    columns = ["tend_follow_-m-_own", "rsp_kill_ours_far_converged"]
    rigid = np.full(1001, -0.012)
    rigid[-1] = 6.5
    varied = np.linspace(-1.0, 1.0, 1001)
    vectors = {"grid": {"MIDDLE": np.stack([rigid, varied])}}

    # when
    reading = _reading(vectors, "MIDDLE", columns, np.array([6.5, 0.9]), np.array([0.3, 0.1]))

    # then
    assert [entry["cell"] for entry in reading["distinctive"]] == ["rsp_kill_ours_far_converged"]
    assert reading["distinctive"][0]["percentile"] == 94.9
