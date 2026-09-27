import gc
import json

import numpy as np
import pandas as pd

from synergy.config import Settings
from synergy.ml import interaction
from synergy.ml.interaction import player_styles, standardise, standardised, style_matrix


def _stream(settings: Settings, matches: int) -> None:
    np.savez(
        settings.processed_dir / "stream.npz",
        match_id=np.array([f"m{i}" for i in range(matches)], dtype=object),
        seat_puuid=np.array([[f"p{(i + seat) % 7}" for seat in range(10)] for i in range(matches)], dtype=object),
        seat_side=np.array([[1] * 5 + [0] * 5] * matches),
    )


def test_the_style_matrix_is_built_once_then_read_from_disk_until_a_source_changes(tmp_path, monkeypatch):
    # given
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    _stream(settings, 3)
    table = pd.DataFrame({"match_id": ["m0", "m1", "m2"], "puuid": ["p0", "p1", "p2"], "x": [1.0, 2.0, 3.0], "y": [4.0, np.nan, 6.0]})
    calls = []

    def load(_):
        calls.append(1)
        return table

    (settings.processed_dir / "a.parquet").write_bytes(b"a")
    monkeypatch.setattr(interaction, "_style_loaders", lambda: ((load, ["x", "y"], "a.parquet"), (lambda _: pd.DataFrame(), ["z"], "b.parquet")))

    # when
    first = style_matrix(settings)
    columns, values = list(first["columns"]), np.array(first["encoding"])
    second = style_matrix(settings)
    reused = isinstance(second["encoding"], np.memmap)
    del first, second
    gc.collect()
    (settings.processed_dir / "a.parquet").write_bytes(b"changed")
    style_matrix(settings)

    # then
    assert len(calls) == 2 and reused
    assert columns == ["x", "y"] and values.shape == (3, 10, 2)
    assert values[0, 0].tolist() == [1.0, 4.0]
    assert np.isnan(values[1, 0, 1]) and np.isnan(values[0, 1]).all()


def test_the_standardised_buffer_matches_standardising_in_memory_and_is_reused(tmp_path):
    # given
    rng = np.random.default_rng(0)
    encoding = rng.normal(size=(40, 10, 3)).astype(np.float32)
    encoding[rng.random((40, 10, 3)) < 0.1] = np.nan
    keep = rng.random(40) > 0.2
    fit = rng.permutation(int(keep.sum()))[: int(keep.sum() * 0.8)]
    folder = tmp_path / "basis"
    folder.mkdir()
    (folder / "style.json").write_text(json.dumps({"sources": [["s", 1, 2]], "columns": ["a", "b", "c"]}), encoding="utf-8")

    # when
    reduced, centre, spread = standardised(folder, encoding, keep, fit)
    values = np.array(reduced)
    again, _, _ = standardised(folder, encoding, keep, fit)
    expected, expected_centre, expected_spread = standardise(encoding[keep], fit)

    # then
    assert np.allclose(values, expected, atol=1e-5)
    assert np.allclose(centre, expected_centre, atol=1e-6) and np.allclose(spread, expected_spread, atol=1e-6)
    assert isinstance(again, np.memmap) and np.array_equal(np.array(again), values)


def test_player_styles_average_every_seat_of_a_player_in_a_position():
    # given
    rng = np.random.default_rng(1)
    reduced = rng.normal(size=(30, 10, 4)).astype(np.float32)
    puuids = np.array([[f"p{rng.integers(6)}" for _ in range(10)] for _ in range(30)], dtype=object)
    positions = np.array([["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2 for _ in range(30)], dtype=object)
    columns = ["a", "b", "c", "d"]

    # when
    styles = player_styles({"reduced": reduced, "seat_puuid": puuids, "seat_position": positions}, columns)

    # then
    flat = pd.DataFrame(reduced.reshape(-1, 4), columns=columns).assign(puuid=puuids.ravel(), position=positions.ravel())
    expected = flat.groupby(["puuid", "position"])
    assert list(styles.index.names) == ["puuid", "position"]
    assert styles.index.equals(expected.size().index)
    assert np.allclose(styles[columns].to_numpy(), expected[columns].mean().to_numpy(), atol=1e-6)
    assert styles["seats"].tolist() == expected.size().tolist()
