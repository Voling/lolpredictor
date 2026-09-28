import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from synergy import queue
from synergy.config import Settings
from synergy.features.regions import REGIONS
from synergy.ml.movement import combine_counts


def _echo(settings, value, shard):
    return [value * 2, shard, settings.feature_minutes, str(settings.data_dir)]


def test_jobs_sent_through_the_queue_come_back_in_call_order_with_the_callers_settings(monkeypatch, tmp_path):
    # given
    monkeypatch.setitem(queue.JOBS, "echo", f"{__name__}:_echo")
    settings = Settings(data_dir=tmp_path, feature_minutes=17)

    # when
    found = queue.fan_out("echo", settings, [(3, 0), (5, 1), (7, 2)])

    # then
    where = str(tmp_path.resolve())
    assert [list(item) for item in found] == [[6, 0, 17, where], [10, 1, 17, where], [14, 2, 17, where]]


def test_a_worker_runs_each_job_in_its_own_process_through_request_and_answer_files(monkeypatch, tmp_path):
    # given
    monkeypatch.setitem(queue.JOBS, "echo", f"{__name__}:_echo")
    monkeypatch.setitem(queue.app.conf, "task_always_eager", False)
    launched = []

    def child(command, capture_output, text):
        launched.append(command)
        queue.answer_request(Path(command[-2]), Path(command[-1]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(queue.subprocess, "run", child)
    values = queue.settings_fields(Settings(data_dir=tmp_path, feature_minutes=19))

    # when
    found = queue.job("echo", values, [4, 2])

    # then
    assert found == [8, 2, 19, str(tmp_path.resolve())]
    assert launched[0][1:3] == ["-m", "synergy.queue"]


def test_a_job_whose_process_fails_raises_with_the_end_of_its_error(monkeypatch, tmp_path):
    # given
    monkeypatch.setitem(queue.app.conf, "task_always_eager", False)
    monkeypatch.setattr(queue.subprocess, "run", lambda command, capture_output, text: subprocess.CompletedProcess(command, 1, "", "Traceback\nValueError: boom"))
    values = queue.settings_fields(Settings(data_dir=tmp_path))

    # when
    with pytest.raises(RuntimeError) as failed:
        queue.job("movement", values, [[], 0])

    # then
    assert "movement job failed with exit 1" in str(failed.value) and "ValueError: boom" in str(failed.value)


def test_settings_cross_the_queue_as_json_without_the_riot_key(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path, riot_api_key="secret", feature_minutes=20)

    # when
    values = json.loads(json.dumps(queue.settings_fields(settings)))
    back = queue.settings_from(values)

    # then
    assert "riot_api_key" not in values
    assert back.data_dir == tmp_path.resolve() and back.feature_minutes == 20 and back.database_url == settings.database_url


def test_a_job_outside_the_allowlist_is_refused():
    # given
    settings = Settings()

    # when
    with pytest.raises(KeyError) as refused:
        queue.fan_out("os:system", settings, [("echo hi",)])

    # then
    assert "unknown job" in str(refused.value)


def test_movement_shards_on_disk_add_up_per_player_in_first_seen_order(tmp_path):
    # given
    size = len(REGIONS)
    first, second, empty = (tmp_path / f"movement_counts.{index:03d}.npz" for index in range(3))
    np.savez(first, players=np.array(["a", "b"]), train=np.ones((2, size, size), np.int32), test=np.zeros((2, size, size), np.int32))
    np.savez(empty, players=np.array([], dtype=str), train=np.zeros((0, size, size), np.int32), test=np.zeros((0, size, size), np.int32))
    np.savez(second, players=np.array(["b", "c"]), train=np.full((2, size, size), 2, np.int32), test=np.ones((2, size, size), np.int32))

    # when
    train, test, players = combine_counts([first, empty, second])

    # then
    assert players == ["a", "b", "c"]
    assert train[:, 0, 0].tolist() == [1, 3, 2] and test[:, 0, 0].tolist() == [0, 1, 1]
    assert not first.exists() and not second.exists() and not empty.exists()
