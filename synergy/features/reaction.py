import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import Settings, get_settings
from .cells import cell_block, combine_shares, evidence_share

TABLE = "reaction.parquet"
EVIDENCE = "evidence_reaction.parquet"
DISTANCE_BANDS = ((2.5, "near"), (5.0, "mid"), (np.inf, "far"))
TRIGGERS = ("kill", "plate", "objective", "building")
RESPONSES = ("converged", "held", "left", "present", "absent")
OBJECTIVES = ("DRAGON", "HORDE")
OBJECTIVE_RESPONSES = ("died", "fought", "committed", "rotated", "approached", "absent")
MINUTE_BANDS = ((5.0, "early"), (10.0, "mid"), (np.inf, "late"))
WARD_ZONES = ("lane_own_side", "lane_middle", "lane_enemy_side", "own_jungle", "enemy_jungle", "river", "other")
OPENING_BANDS = ((3.0, "by3"), (5.0, "by5"), (8.0, "by8"), (15.0, "late"), (np.inf, "never"))
SIDES = ("crossed", "stayed")

RESPONSE_SITUATIONS = [f"{t}_{side}_{band}" for t in TRIGGERS for side in ("ours", "theirs") for _, band in DISTANCE_BANDS]
OBJECTIVE_SITUATIONS = [f"{o}_{side}_{band}" for o in OBJECTIVES for side in ("ours", "theirs") for _, band in DISTANCE_BANDS]
WARD_SITUATIONS = [band for _, band in MINUTE_BANDS]
OPENING_SITUATIONS = ["gank", "invade"]
OPENING_OUTCOMES = [band for _, band in OPENING_BANDS]

REACTION_COLUMNS = (
    [f"rsp_{s}_{o}" for s in RESPONSE_SITUATIONS for o in RESPONSES]
    + [f"obj_{s}_{o}" for s in OBJECTIVE_SITUATIONS for o in OBJECTIVE_RESPONSES]
    + [f"ward_{s}_{o}" for s in WARD_SITUATIONS for o in WARD_ZONES]
    + [f"jgl_{s}_{o}" for s in OPENING_SITUATIONS for o in OPENING_OUTCOMES]
    + [f"jgl_sides_{o}" for o in SIDES]
)


RESPONSE_READ = ["match_id", "puuid", "trigger", "ours", "is_actor", "is_victim", "approach", "present", "converged", "left_after", "held_ground"]
OBJECTIVE_READ = [
    "match_id", "puuid", "objective", "ours", "o_approach_distance", "o_died", "o_fought", "o_committed", "o_rotated_in", "o_approaching",
]


def _responses(settings: Settings) -> pd.DataFrame:
    table = pq.read_table(
        settings.processed_dir / "event_responses.parquet",
        columns=RESPONSE_READ,
        filters=[("is_actor", "==", 0), ("is_victim", "==", 0), ("trigger", "in", list(TRIGGERS))],
        use_pandas_metadata=True,
    )
    return table.to_pandas(split_blocks=True, self_destruct=True)


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
    return pd.DataFrame({"match_id": column("match_id"), "puuid": column("puuid"), "situation": situation, "outcome": outcome, "count": 1.0})


def objective_counts(objectives: pd.DataFrame) -> pd.DataFrame:
    rows = objectives[objectives["objective"].isin(OBJECTIVES)]
    side = np.where(rows["ours"] == 1, "ours", "theirs")
    situation = rows["objective"].astype(str) + "_" + side + "_" + _band(rows["o_approach_distance"], DISTANCE_BANDS).to_numpy()
    outcome = np.select(
        [rows["o_died"] > 0, rows["o_fought"] > 0, rows["o_committed"] > 0, rows["o_rotated_in"] > 0, rows["o_approaching"] > 0],
        ["died", "fought", "committed", "rotated", "approached"],
        default="absent",
    )
    return pd.DataFrame({"match_id": rows["match_id"].to_numpy(), "puuid": rows["puuid"].to_numpy(), "situation": situation.to_numpy(), "outcome": outcome, "count": 1.0})


def _ward_zone(zone: pd.Series) -> np.ndarray:
    z = zone.fillna("").astype(str)
    return np.select(
        [z.str.endswith("_OWN") & z.str.startswith("LANE"), z.str.endswith("_NEUTRAL"), z.str.endswith("_ENEMY") & z.str.startswith("LANE"),
         z.str.startswith("JUNGLE_OWN"), z.str.startswith("JUNGLE_ENEMY"), z.str.startswith("RIVER")],
        ["lane_own_side", "lane_middle", "lane_enemy_side", "own_jungle", "enemy_jungle", "river"],
        default="other",
    )


def ward_counts(wards: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": wards["match_id"].to_numpy(), "puuid": wards["puuid"].to_numpy(),
        "situation": _band(wards["minute"], MINUTE_BANDS).to_numpy(), "outcome": _ward_zone(wards["zone"]), "count": 1.0,
    })


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
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"])
    openings = jungle_counts(pd.read_parquet(settings.processed_dir / "jungle_openings.parquet"))
    sources = (
        ("rsp", lambda: response_counts(_responses(settings)), RESPONSE_SITUATIONS, list(RESPONSES)),
        ("obj", lambda: objective_counts(pd.read_parquet(settings.processed_dir / "objectives.parquet", columns=OBJECTIVE_READ)),
         OBJECTIVE_SITUATIONS, list(OBJECTIVE_RESPONSES)),
        ("ward", lambda: ward_counts(pd.read_parquet(settings.processed_dir / "wards.parquet", columns=["match_id", "puuid", "minute", "zone"])), WARD_SITUATIONS, list(WARD_ZONES)),
        ("jgl", lambda: openings[openings["situation"] != "sides"], OPENING_SITUATIONS, OPENING_OUTCOMES),
        ("jgl", lambda: openings[openings["situation"] == "sides"], ["sides"], list(SIDES)),
    )
    keys, values, report, shares = None, None, {}, []
    for prefix, load, situations, outcomes in sources:
        counts = load()
        block, found = cell_block(counts, seats, situations, outcomes, prefix)
        if values is None:
            keys = block[["match_id", "puuid"]]
            values = np.full((len(block), len(REACTION_COLUMNS)), np.nan)
        values[:, [REACTION_COLUMNS.index(column) for column in block.columns[2:]]] = block.iloc[:, 2:].to_numpy(dtype=float)
        del block
        shares.append((len(situations) * len(outcomes), evidence_share(counts, seats, situations, {s: found[s]["kappa"] for s in situations})))
        report[f"{prefix}:{situations[0]}" if prefix in report else prefix] = {"rows": int(counts["count"].sum()), "situations": len(situations), "outcomes": len(outcomes),
                          "kappa_range": [min(v["kappa"] for v in found.values()), max(v["kappa"] for v in found.values())]}
        del counts
    table = pd.DataFrame(values, columns=REACTION_COLUMNS)
    table.insert(0, "puuid", keys["puuid"].to_numpy())
    table.insert(0, "match_id", keys["match_id"].to_numpy())
    table.to_parquet(settings.processed_dir / TABLE, index=False)
    evidence = combine_shares(shares)
    evidence.to_parquet(settings.processed_dir / EVIDENCE, index=False)
    report["columns"] = len(REACTION_COLUMNS)
    report["rows"] = int(len(table))
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
