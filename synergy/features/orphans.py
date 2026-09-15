import numpy as np
import pandas as pd

from ..config import Settings, get_settings

WARD_COLUMNS = ["in_enemy_jungle", "in_own_jungle", "in_river", "warded_before_invade", "warded_near_next_gank"]
OBJECTIVE_COLUMNS = ["o_present", "o_approaching", "o_committed", "o_rotated_in", "o_left_after", "o_fought", "o_died"]
RESPONSE_COLUMNS = ["is_actor", "is_victim", "present", "nearby", "converged", "left_after", "held_ground", "latency"]
JUNGLE_COLUMNS = ["first_invade_minute", "first_gank_minute", "jungle_cs_at_3", "jungle_cs_at_5", "crossed_sides"]

ORPHAN_COLUMNS = (
    [f"o_ward_{name}" for name in WARD_COLUMNS]
    + ["o_ward_rate"]
    + [f"o_obj_{name}" for name in OBJECTIVE_COLUMNS]
    + [f"o_rsp_{name}" for name in RESPONSE_COLUMNS]
    + [f"o_jgl_{name}" for name in JUNGLE_COLUMNS]
)
TABLE = "orphan_features.parquet"


def _per_player(settings: Settings, name: str, columns: list[str], prefix: str) -> pd.DataFrame:
    path = settings.processed_dir / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid"])
    frame = pd.read_parquet(path, columns=["match_id", "puuid", *columns])
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    grouped = frame.groupby(["match_id", "puuid"])[columns].mean()
    grouped.columns = [f"{prefix}{column}" for column in columns]
    return grouped.reset_index()


def build_orphan_features(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    seats = pd.read_parquet(
        settings.processed_dir / "participations.parquet", columns=["match_id", "puuid"]
    )

    wards = pd.read_parquet(
        settings.processed_dir / "wards.parquet", columns=["match_id", "puuid", *WARD_COLUMNS]
    )
    for column in WARD_COLUMNS:
        wards[column] = pd.to_numeric(wards[column], errors="coerce").astype(float)
    placed = wards.groupby(["match_id", "puuid"]).size().rename("o_ward_rate")
    ward_rates = wards.groupby(["match_id", "puuid"])[WARD_COLUMNS].mean()
    ward_rates.columns = [f"o_ward_{column}" for column in WARD_COLUMNS]
    ward_block = ward_rates.join(placed).reset_index()

    blocks = [
        seats,
        ward_block,
        _per_player(settings, "objectives", OBJECTIVE_COLUMNS, "o_obj_"),
        _per_player(settings, "event_responses", RESPONSE_COLUMNS, "o_rsp_"),
        _per_player(settings, "jungle_openings", JUNGLE_COLUMNS, "o_jgl_"),
    ]
    table = blocks[0]
    for block in blocks[1:]:
        table = table.merge(block, on=["match_id", "puuid"], how="left")
    for column in ORPHAN_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    table = table[["match_id", "puuid", *ORPHAN_COLUMNS]]

    values = table[ORPHAN_COLUMNS].to_numpy(dtype=float)
    known = ~np.isnan(values)
    filled = np.where(known, values, 0.0)
    frame = pd.DataFrame(filled, columns=ORPHAN_COLUMNS)
    frame["puuid"] = table.puuid.to_numpy()
    seen = pd.DataFrame(known.astype(float), columns=ORPHAN_COLUMNS)
    seen["puuid"] = table.puuid.to_numpy()
    totals = frame.groupby("puuid")[ORPHAN_COLUMNS].transform("sum").to_numpy()
    counts = seen.groupby("puuid")[ORPHAN_COLUMNS].transform("sum").to_numpy()
    others = counts - known
    loo = np.divide(
        totals - filled, others, out=np.full(values.shape, np.nan), where=others > 0
    )
    table[ORPHAN_COLUMNS] = loo
    table.to_parquet(settings.processed_dir / TABLE, index=False)
    return {
        "rows": int(len(table)),
        "columns": len(ORPHAN_COLUMNS),
        "leave_one_out": True,
        "coverage": {
            column: round(float(table[column].notna().mean()), 4)
            for column in ("o_ward_rate", "o_obj_o_present", "o_rsp_present", "o_jgl_crossed_sides")
        },
    }


def load_orphan_features(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *ORPHAN_COLUMNS])
    return pd.read_parquet(path)
