import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import Settings, get_settings
from .objectives import SIDES as OBJECTIVE_SIDES
from .cells import SeatIndex, buffer, combine_shares, dense_counts, evidence_shares, fit_cells, moment_kappa, split_columns, write_cells
from .fits import REACTION_FITS, reaction_sources, save_cells

STATES = [f"{band}_{gold}" for band in ("early", "late") for gold in ("behind", "even", "ahead")]
UNSTATED = "early_even"
LATE_MINUTE = 10.0
LANE_GOLD = 500.0
TABLE = "reaction.parquet"
PIECE_ROWS = 2_000_000
EVIDENCE = "evidence_reaction.parquet"
DISTANCE_BANDS = ((2.5, "near"), (5.0, "mid"), (np.inf, "far"))
TRIGGERS = ("kill", "plate", "objective", "building")
RESPONSES = ("converged", "held", "left", "present", "absent")
OBJECTIVES = ("DRAGON", "HORDE", "RIFTHERALD")
OBJECTIVE_RESPONSES = ("died", "fought", "committed", "rotated", "approached", "absent")
MINUTE_BANDS = ((5.0, "early"), (10.0, "mid"), (np.inf, "late"))
WARD_ZONES = ("lane_own_side", "lane_middle", "lane_enemy_side", "own_jungle", "enemy_jungle", "river", "other")
ZONE_COLUMNS = [f"in_{zone}" for zone in WARD_ZONES]
OPENING_BANDS = ((3.0, "by3"), (5.0, "by5"), (8.0, "by8"), (15.0, "late"), (np.inf, "never"))
SIDES = ("crossed", "stayed")

RESPONSE_SITUATIONS = [f"{t}_{side}_{band}" for t in TRIGGERS for side in ("ours", "theirs") for _, band in DISTANCE_BANDS]
OBJECTIVE_SITUATIONS = [f"{o}_{side}_{band}" for o in OBJECTIVES for side in OBJECTIVE_SIDES for _, band in DISTANCE_BANDS]
WARD_SITUATIONS = [band for _, band in MINUTE_BANDS]
OPENING_SITUATIONS = ["gank", "invade"]
OPENING_OUTCOMES = [band for _, band in OPENING_BANDS]
GOLD_TRIGGERS = ("kill", "plate")
GOLD_SPLIT = {
    "prefix": "rspg",
    "situations": [situation for situation in RESPONSE_SITUATIONS if situation.split("_")[0] in GOLD_TRIGGERS],
    "groups": {gold: [state for state in STATES if state.endswith(f"_{gold}")] for gold in ("behind", "ahead")},
}
GOLD_COLUMNS = split_columns(GOLD_SPLIT, list(RESPONSES))

REACTION_COLUMNS = (
    [f"rsp_{s}_{o}" for s in RESPONSE_SITUATIONS for o in RESPONSES]
    + GOLD_COLUMNS
    + [f"obj_{s}_{o}" for s in OBJECTIVE_SITUATIONS for o in OBJECTIVE_RESPONSES]
    + [f"ward_{s}_{o}" for s in WARD_SITUATIONS for o in WARD_ZONES]
    + [f"jgl_{s}_{o}" for s in OPENING_SITUATIONS for o in OPENING_OUTCOMES]
    + [f"jgl_sides_{o}" for o in SIDES]
)


RESPONSE_READ = ["match_id", "puuid", "trigger", "ours", "is_actor", "is_victim", "approach", "present", "converged", "left_after", "held_ground", "state"]
OBJECTIVE_READ = [
    "match_id", "puuid", "objective", "ours", "side", "o_approach_distance", "o_died", "o_fought", "o_committed", "o_rotated_in", "o_approaching", "state",
]


def chance_state(minute: float, lead: float) -> str:
    gold = "behind" if lead < -LANE_GOLD else "ahead" if lead > LANE_GOLD else "even"
    return f"{'early' if minute < LATE_MINUTE else 'late'}_{gold}"


def _pieces(path, columns: list[str]):
    source = pq.ParquetFile(path)
    present = [column for column in columns if column in source.schema_arrow.names]
    for batch in source.iter_batches(batch_size=PIECE_ROWS, columns=present):
        yield batch.to_pandas()


def _responses(settings: Settings):
    for piece in _pieces(settings.processed_dir / "event_responses.parquet", RESPONSE_READ):
        yield piece[(piece["is_actor"] == 0) & (piece["is_victim"] == 0) & piece["trigger"].isin(TRIGGERS)]


def _band_codes(values: np.ndarray, bands) -> np.ndarray:
    return np.searchsorted([edge for edge, _ in bands], values, side="right").clip(max=len(bands) - 1)


def _band(values: pd.Series, bands) -> pd.Series:
    names = [name for _, name in bands]
    return pd.Series(np.array(names)[_band_codes(values.to_numpy(dtype=float), bands)], index=values.index)


def response_counts(responses: pd.DataFrame) -> pd.DataFrame:
    kept = ((responses["is_actor"] == 0) & (responses["is_victim"] == 0) & responses["trigger"].isin(TRIGGERS)).to_numpy()

    def column(name: str) -> np.ndarray:
        return responses[name].to_numpy()[kept]

    bands = [band for _, band in DISTANCE_BANDS]
    names = np.array([f"{t}_{side}_{band}" for t in TRIGGERS for side in ("ours", "theirs") for band in bands], dtype=object)
    trigger = pd.Index(TRIGGERS).get_indexer(column("trigger")).astype(np.int64)
    theirs = (column("ours") != 1).astype(np.int64)
    situation = names[(trigger * 2 + theirs) * len(bands) + _band_codes(column("approach").astype(float), DISTANCE_BANDS)]
    outcome = np.array(["converged", "held", "left", "present", "absent"], dtype=object)[
        np.select(
            [column("converged") > 0, column("held_ground") > 0, column("left_after") > 0, column("present") > 0],
            [0, 1, 2, 3],
            default=4,
        )
    ]
    state = column("state") if "state" in responses.columns else np.full(int(kept.sum()), UNSTATED, dtype=object)
    return pd.DataFrame({"match_id": column("match_id"), "puuid": column("puuid"), "situation": situation, "state": state, "outcome": outcome, "count": 1.0})


def objective_counts(objectives: pd.DataFrame) -> pd.DataFrame:
    rows = objectives[objectives["objective"].isin(OBJECTIVES)]
    side = rows["side"].astype(str).to_numpy() if "side" in rows.columns else np.where(rows["ours"] == 1, "ours", "theirs")
    situation = rows["objective"].astype(str) + "_" + side + "_" + _band(rows["o_approach_distance"], DISTANCE_BANDS).to_numpy()
    state = rows["state"].astype(str).to_numpy() if "state" in rows.columns else np.full(len(rows), UNSTATED, dtype=object)
    outcome = np.select(
        [rows["o_died"] > 0, rows["o_fought"] > 0, rows["o_committed"] > 0, rows["o_rotated_in"] > 0, rows["o_approaching"] > 0],
        ["died", "fought", "committed", "rotated", "approached"],
        default="absent",
    )
    return pd.DataFrame(
        {"match_id": rows["match_id"].to_numpy(), "puuid": rows["puuid"].to_numpy(), "situation": situation.to_numpy(), "state": state, "outcome": outcome, "count": 1.0}
    )


def ward_counts(wards: pd.DataFrame) -> pd.DataFrame:
    zones = len(WARD_ZONES)
    return pd.DataFrame({
        "match_id": np.repeat(wards["match_id"].to_numpy(), zones), "puuid": np.repeat(wards["puuid"].to_numpy(), zones),
        "situation": np.repeat(_band(wards["minute"], MINUTE_BANDS).to_numpy(), zones), "outcome": np.tile(np.array(WARD_ZONES, dtype=object), len(wards)),
        "count": wards[ZONE_COLUMNS].to_numpy(dtype=float).ravel(),
    })


def ward_kappa(counts: np.ndarray, index: SeatIndex) -> list[float]:
    kappas = []
    for step in range(counts.shape[1]):
        wards = np.asarray(counts[:, step], dtype=float).sum(axis=(1, 2))
        played = wards > 0
        per_game = float(wards[played].mean()) if played.any() else 1.0
        kappas.append(moment_kappa(counts, index, step, by_position=True) * per_game)
    return kappas


def jungle_counts(openings: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for situation, column in (("gank", "first_gank_minute"), ("invade", "first_invade_minute")):
        parts.append(pd.DataFrame({
            "match_id": openings["match_id"].to_numpy(), "puuid": openings["puuid"].to_numpy(),
            "situation": situation, "outcome": _band(openings[column].fillna(np.inf), OPENING_BANDS).to_numpy(), "count": 1.0,
        }))
    parts.append(pd.DataFrame({
        "match_id": openings["match_id"].to_numpy(), "puuid": openings["puuid"].to_numpy(),
        "situation": "sides", "outcome": np.where(openings["crossed_sides"].astype(bool), "crossed", "stayed"), "count": 1.0,
    }))
    return pd.concat(parts, ignore_index=True)


def build_reaction(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    processed = settings.processed_dir
    index = SeatIndex(pd.read_parquet(processed / "participations.parquet", columns=["match_id", "puuid", "position"]))
    parts, paths, report, shares = [], [], {}, []
    for number, (prefix, load, situations, outcomes, states, split, kappa) in enumerate(reaction_sources(settings)):
        path = processed / "buffers" / f"reaction_counts.{number}.npy"
        counts = buffer(path, index, situations, outcomes, states)
        for frame in load():
            dense_counts(frame, index, situations, outcomes, out=counts, states=states)
        counts.flush()
        fit = fit_cells(
            counts, index, situations, outcomes, prefix, scale=settings.cell_prior_scale, kappa=None if kappa is None else kappa(counts, index), states=states, split=split
        )
        save_cells(settings.model_dir / REACTION_FITS[number], fit, index.positions)
        parts.append((fit, counts))
        paths.append(path)
        shares.append((len(situations) * len(outcomes), evidence_shares(fit, index)))
        kappas = [entry["kappa"] for entry in fit.report.values()]
        report[f"{prefix}:{situations[0]}" if prefix in report else prefix] = {
            "rows": int(sum(entry["rows"] for entry in fit.report.values())),
            "situations": len(situations),
            "states": len(states),
            "outcomes": len(outcomes),
            "kappa_range": [min(kappas), max(kappas)],
        }
    write_cells(processed / TABLE, parts, index, REACTION_COLUMNS)
    del parts, counts
    for path in paths:
        path.unlink(missing_ok=True)
    evidence = combine_shares(shares)
    evidence.to_parquet(processed / EVIDENCE, index=False)
    report["columns"] = len(REACTION_COLUMNS)
    report["rows"] = len(index)
    report["evidence_share_median"] = round(float(evidence["share"].median()), 3)
    return report


def load_reaction(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *REACTION_COLUMNS])
    table = pd.read_parquet(path)
    if any(column not in table.columns for column in REACTION_COLUMNS):
        return pd.DataFrame(columns=["match_id", "puuid", *REACTION_COLUMNS])
    return table
