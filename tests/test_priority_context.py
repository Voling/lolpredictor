import numpy as np
import pandas as pd

from synergy.features.priority import CONTEXT_COLUMNS, priority_context_from


def test_context_carries_each_lane_of_the_team_and_junglers_take_the_best_lane():
    # given
    track = pd.DataFrame(
        {
            "match_id": ["m"] * 7,
            "puuid": ["top", "top", "mid", "adc", "sup", "jng", "top"],
            "tick": [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 1.0],
            "prio_filtered": [1.0, 0.0, 0.2, 0.6, 0.8, None, 0.9],
        }
    )
    seats = pd.DataFrame(
        {
            "match_id": ["m"] * 5,
            "puuid": ["top", "mid", "adc", "sup", "jng"],
            "team_id": [100] * 5,
            "position": ["TOP", "MIDDLE", "BOTTOM", "UTILITY", "JUNGLE"],
        }
    )

    # when
    out = priority_context_from(track, seats).set_index(["puuid", "minute"])

    # then
    assert list(out.columns) == ["match_id", *CONTEXT_COLUMNS]
    assert np.isclose(out.loc[("top", 0), "top_priority"], 0.5)
    assert np.isclose(out.loc[("top", 0), "mid_priority"], 0.2)
    assert np.isclose(out.loc[("top", 0), "bot_priority"], 0.7)
    assert np.isclose(out.loc[("top", 0), "lane_priority"], 0.5)
    assert np.isclose(out.loc[("sup", 0), "lane_priority"], 0.7)
    assert np.isclose(out.loc[("jng", 0), "lane_priority"], 0.7)
    assert np.isclose(out.loc[("jng", 1), "top_priority"], 0.9)
    assert np.isclose(out.loc[("jng", 1), "mid_priority"], 0.9)


def test_the_context_read_row_group_by_row_group_matches_reading_the_whole_track(tmp_path):
    # given
    import pyarrow as pa
    import pyarrow.parquet as pq

    from synergy.config import Settings
    from synergy.features.priority import TRACK, priority_context

    rng = np.random.default_rng(5)
    seats = pd.DataFrame(
        {
            "match_id": ["m1"] * 10 + ["m2"] * 10,
            "puuid": [f"p{seat}" for seat in range(10)] * 2,
            "team_id": ([100] * 5 + [200] * 5) * 2,
            "position": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 4,
        }
    )
    ticks = np.round(np.arange(0.0, 3.0, 1.0 / 6.0), 6)
    track = pd.DataFrame(
        {
            "match_id": np.repeat(seats["match_id"].to_numpy(), len(ticks)),
            "puuid": np.repeat(seats["puuid"].to_numpy(), len(ticks)),
            "tick": np.tile(ticks, len(seats)),
            "prio_filtered": np.where(rng.random(len(seats) * len(ticks)) > 0.2, rng.random(len(seats) * len(ticks)), np.nan),
        }
    )
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    seats.to_parquet(settings.processed_dir / "participations.parquet", index=False)
    with pq.ParquetWriter(settings.processed_dir / TRACK, pa.Schema.from_pandas(track, preserve_index=False)) as writer:
        for piece in np.array_split(np.arange(len(track)), 3):
            writer.write_table(pa.Table.from_pandas(track.iloc[piece], preserve_index=False))

    # when
    streamed = priority_context(settings)
    whole = priority_context_from(track, seats)

    # then
    key = ["match_id", "puuid", "minute"]
    assert pq.ParquetFile(settings.processed_dir / TRACK).num_row_groups == 3
    left = streamed.sort_values(key).reset_index(drop=True)
    right = whole.sort_values(key).reset_index(drop=True)
    assert left[key].equals(right[key])
    assert np.allclose(left.drop(columns=key).to_numpy(dtype=float), right.drop(columns=key).to_numpy(dtype=float), equal_nan=True)
