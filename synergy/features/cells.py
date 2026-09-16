import numpy as np
import pandas as pd

from ..ml.movement import best_kappa
from .positions import KEY

FALLBACK_KAPPA = 8.0
MIN_PLAYERS = 50


def _kappa(counts: np.ndarray, world: np.ndarray) -> float:
    totals = counts.sum(axis=1)
    seen = totals > 0
    if seen.sum() < MIN_PLAYERS:
        return FALLBACK_KAPPA
    return float(best_kappa(counts[seen], totals[seen], world[seen]))


def evidence_share(counts: pd.DataFrame, seats: pd.DataFrame, situations: list[str], kappas: dict[str, float]) -> pd.DataFrame:
    seated = seats[["match_id", *KEY]].drop_duplicates(["match_id", "puuid"])
    everyone = seated[KEY].drop_duplicates().set_index(KEY)
    exposure = (
        counts[counts["situation"].isin(situations)]
        .merge(seated, on=["match_id", "puuid"], how="inner")
        .groupby([*KEY, "situation"])["count"].sum().unstack("situation")
        .reindex(index=everyone.index, columns=situations, fill_value=0.0).fillna(0.0)
        .to_numpy(dtype=float)
    )
    kappa = np.array([kappas[situation] for situation in situations], dtype=float)
    weight = exposure / (exposure + kappa[None, :])
    typical = exposure.mean(axis=0)
    share = (weight * typical[None, :]).sum(axis=1) / max(float(typical.sum()), 1e-9)
    return pd.DataFrame({
        "puuid": everyone.index.get_level_values("puuid"),
        "position": everyone.index.get_level_values("position"),
        "share": share,
    })


def combine_shares(parts: list[tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    total = sum(weight for weight, _ in parts)
    merged = None
    for weight, frame in parts:
        scaled = frame.set_index(KEY)["share"] * (weight / total)
        merged = scaled if merged is None else merged.add(scaled, fill_value=0.0)
    return merged.rename("share").reset_index()


def position_worlds(frame: pd.DataFrame, outcomes: int) -> dict[str, np.ndarray]:
    pooled = frame.groupby(level="position").sum()
    return {
        position: (row.to_numpy(dtype=float) + 1.0) / (float(row.sum()) + outcomes)
        for position, row in pooled.iterrows()
    }


def cell_block(
    counts: pd.DataFrame,
    seats: pd.DataFrame,
    situations: list[str],
    outcomes: list[str],
    prefix: str,
) -> tuple[pd.DataFrame, dict]:
    columns = [f"{prefix}_{situation}_{outcome}" for situation in situations for outcome in outcomes]
    table = (
        counts.groupby(["match_id", "puuid", "situation", "outcome"])["count"].sum().unstack("outcome")
        .reindex(columns=outcomes, fill_value=0.0).fillna(0.0)
    )
    out = seats[["match_id", *KEY]].drop_duplicates(["match_id", "puuid"]).set_index(["match_id", "puuid"])
    positions = out["position"].to_numpy()
    who = pd.MultiIndex.from_arrays([out.index.get_level_values("puuid"), positions], names=KEY)
    blocks, report = [], {}
    for situation in situations:
        here = table.xs(situation, level="situation") if situation in table.index.get_level_values("situation") else table.iloc[0:0].droplevel("situation")
        values = here.reindex(out.index, fill_value=0.0).to_numpy(dtype=float)
        frame = pd.DataFrame(values, index=who)
        per_who = frame.groupby(level=KEY).sum()
        worlds = position_worlds(frame, len(outcomes))
        kappa = _kappa(
            per_who.to_numpy(dtype=float),
            np.stack([worlds[position] for position in per_who.index.get_level_values("position")]),
        )
        world = np.stack([worlds[position] for position in positions])
        others = per_who.reindex(who).to_numpy(dtype=float) - values
        exposure = others.sum(axis=1, keepdims=True)
        posterior = (others + kappa * world) / (exposure + kappa)
        blocks.append(pd.DataFrame(posterior, index=out.index, columns=[f"{prefix}_{situation}_{o}" for o in outcomes]))
        report[situation] = {
            "rows": int(values.sum()),
            "kappa": round(kappa, 2),
            "world": {position: [round(float(v), 4) for v in row] for position, row in worlds.items()},
        }
    frame = pd.concat(blocks, axis=1).reindex(columns=columns).reset_index()
    return frame, report
