import pandas as pd
import pytest

from synergy.chunks import ChunkWriter


def test_rows_survive_several_flushes(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=10)
    for index in range(45):
        writer.add([{"a": index, "b": float(index) / 2}])
    assert writer.close() == 45
    frame = pd.read_parquet(path)
    assert len(frame) == 45
    assert frame["a"].tolist() == list(range(45))


def test_the_first_chunk_fixes_the_schema_for_the_rest(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    writer.add([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
    writer.add([{"a": 5.0}, {"a": 6.0}])
    writer.add([{"a": 7.0, "b": 8.0, "surprise": 9.0}, {"a": 9.0, "b": 1.0, "surprise": 9.0}])
    writer.close()
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["a", "b"]
    assert frame["a"].tolist() == [1.0, 3.0, 5.0, 6.0, 7.0, 9.0]
    assert frame["b"].isna().tolist() == [False, False, True, True, False, False]


def test_nothing_written_still_leaves_a_readable_file(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    assert writer.close() == 0
    assert pd.read_parquet(path).empty


def test_empty_record_lists_are_ignored(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    writer.add([])
    writer.add([{"a": 1.0}])
    writer.add([])
    assert writer.close() == 1


def test_an_int_column_missing_later_is_filled_with_nulls(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    writer.add([{"a": 1, "n": 10}, {"a": 2, "n": 20}])
    writer.add([{"a": 3}, {"a": 4}])
    writer.close()
    frame = pd.read_parquet(path)
    assert frame["n"].isna().tolist() == [False, False, True, True]
    assert frame["a"].tolist() == [1, 2, 3, 4]


def test_a_column_that_changes_type_is_rejected_loudly(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    writer.add([{"a": 1}, {"a": 2}])
    with pytest.raises(ValueError, match="changed type mid-file"):
        writer.add([{"a": "text"}, {"a": "more"}])


def test_widening_an_int_column_to_floats_is_rejected_rather_than_truncated(tmp_path):
    path = tmp_path / "out.parquet"
    writer = ChunkWriter(path, chunk=2)
    writer.add([{"a": 1}, {"a": 2}])
    with pytest.raises(ValueError, match="changed type mid-file"):
        writer.add([{"a": 3.5}, {"a": 4.5}])


def test_splitting_match_ids_covers_every_id_exactly_once():
    from synergy.features.build import _split

    ids = [f"m{i}" for i in range(23)]
    for parts in (1, 3, 8, 50):
        chunks = _split(ids, parts)
        flat = [x for chunk in chunks for x in chunk]
        assert flat == ids
        assert all(chunk for chunk in chunks)


def test_splitting_an_empty_list_yields_one_empty_chunk():
    from synergy.features.build import _split

    assert _split([], 8) == [[]]


def test_the_build_returns_frames_for_participations_and_pairs_not_counts(tables):
    import pandas as pd

    assert isinstance(tables["participations"], pd.DataFrame)
    assert isinstance(tables["pairs"], pd.DataFrame)
    assert isinstance(tables["policy"], int)
