import json

import numpy as np
import pytest

from synergy.config import Settings
from synergy.pipeline import KEEP_RUNS, list_runs, plan, promote, prune, run_pipeline, served_summary, write_pointer


def _counts(settings):
    return {"matches": 12, "players": 40}


def _working(settings):
    settings.ensure_dirs()
    (settings.model_dir / "interaction_report.json").write_text(
        json.dumps({"matches": 12, "cells_alone": 0.05, "with_interaction": 0.0501, "gain": 0.0001, "interaction_spread": 22.6, "informative": True}),
        encoding="utf-8",
    )
    (settings.model_dir / "seat_report.json").write_text(json.dumps({"held_out": {"TOP": 0.09}}), encoding="utf-8")
    np.savez(settings.model_dir / "interaction_matrix.npz", matrix=np.eye(2))
    (settings.processed_dir / "player_styles.parquet").write_bytes(b"styles")


def test_plan_slices_the_ordered_steps_and_rejects_unknown_ones():
    # given
    wanted = ("stream", "blocks")

    # when
    middle = plan(*wanted)
    tail = plan("mirrored", network=True)

    # then
    assert middle == ["stream", "profiles", "blocks"]
    assert tail == ["mirrored", "scores", "network"]
    with pytest.raises(ValueError):
        plan("blocks", "stream")
    with pytest.raises(ValueError):
        plan("nothing")


def test_a_failing_step_stops_the_run_and_leaves_serving_untouched(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    _working(settings)
    seen = []

    def runner(step, command, log_path):
        seen.append(step)
        log_path.write_text(step, encoding="utf-8")
        return 0 if step != "profiles" else 3

    # when
    manifest = run_pipeline(settings, start="stream", stop="blocks", runner=runner, counts=_counts)

    # then
    assert seen == ["stream", "profiles"]
    assert manifest["status"] == "failed" and [step["exit"] for step in manifest["steps"]] == [0, 3]
    assert manifest["corpus"] == {"matches": 12, "players": 40}
    assert settings.served_run() is None
    assert settings.served_model_dir == settings.model_dir


def test_a_finished_run_is_copied_aside_and_served_atomically(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    _working(settings)

    # when
    manifest = run_pipeline(settings, start="mirrored", runner=lambda step, command, log_path: 0, counts=_counts)
    (settings.processed_dir / "player_styles.parquet").write_bytes(b"rewritten by the next build")

    # then
    assert manifest["status"] == "served" and settings.served_run() == manifest["id"]
    assert settings.served_model_dir == settings.runs_dir / manifest["id"] / "serve" / "models"
    assert (settings.served_processed_dir / "player_styles.parquet").read_bytes() == b"styles"
    assert "models/interaction_matrix.npz" in manifest["served"]["copied"]
    assert "models/synergy_model.pkl" in manifest["served"]["missing"]
    assert manifest["metrics"]["pair"]["gain"] == 0.0001 and manifest["metrics"]["seats"]["held_out"]["TOP"] == 0.09
    assert served_summary(settings)["matches"] == 12 and served_summary(settings)["metrics"]["pair"]["informative"] is True
    assert not settings.pointer_path.with_suffix(".tmp").exists()


def test_old_runs_are_pruned_but_the_served_one_survives(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    _working(settings)
    served = promote(settings, counts=_counts)["id"]
    oldest = "00000000-000000"
    (settings.runs_dir / served).rename(settings.runs_dir / oldest)
    manifest_path = settings.runs_dir / oldest / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace(served, oldest), encoding="utf-8")
    write_pointer(settings, oldest)
    built = [
        run_pipeline(settings, start="scores", runner=lambda *_: 0, counts=_counts, promote_after=False)["id"]
        for _ in range(KEEP_RUNS + 1)
    ]

    # when
    removed = prune(settings, current=oldest)
    runs = list_runs(settings)

    # then
    assert len(set(built)) == KEEP_RUNS + 1
    assert oldest not in removed and (settings.runs_dir / oldest).is_dir()
    assert len(runs) == KEEP_RUNS + 1 and sum(run["current"] for run in runs) == 1
    assert removed == [sorted(built)[0]]
