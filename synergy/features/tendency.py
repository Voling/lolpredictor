import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings, get_settings
from .cells import combine_shares
from .positions import KEY
from .priority import CONTEXT_COLUMNS as CONTEXT, priority_context
from .propensity import COVARIATES, KINDS, _gamma_prior

HELD = 0.65
HALVES = ("own", "away")
PATTERNS = [
    "".join(letter if bit else "-" for letter, bit in zip("tmb", bits))
    for bits in ((a, b, c) for a in (1, 0) for b in (1, 0) for c in (1, 0))
]
CELLS = [f"{pattern}_{half}" for pattern in PATTERNS for half in HALVES]
TENDENCY_COLUMNS = [f"tend_{kind}_{cell}" for kind in KINDS for cell in CELLS]
TABLE = "tendency.parquet"
EVIDENCE = "evidence_tendency.parquet"
MIN_ROWS = 200


def with_priority(opportunities: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return opportunities.assign(**{column: 0.0 for column in CONTEXT})
    lagged = context.assign(minute=context["minute"] + 1)
    joined = opportunities.merge(lagged, on=["match_id", "puuid", "minute"], how="left")
    for column in CONTEXT:
        joined[column] = joined[column].fillna(float(context[column].mean()))
    return joined


def cell_of(top: pd.Series, mid: pd.Series, bot: pd.Series, in_own_half: pd.Series) -> np.ndarray:
    letters = []
    for name, series in zip("tmb", (top, mid, bot)):
        letters.append(np.where(series.to_numpy(dtype=float) >= HELD, name, "-"))
    pattern = np.char.add(np.char.add(letters[0].astype(str), letters[1].astype(str)), letters[2].astype(str))
    half = np.where(in_own_half.to_numpy(dtype=float) > 0, "own", "away")
    return np.char.add(np.char.add(pattern, "_"), half.astype(str))


def leave_one_out_ratio(frame: pd.DataFrame, seats: pd.DataFrame, prior: float, column: str) -> pd.DataFrame:
    whole = frame.groupby([*KEY, "cell"])[["observed", "expected"]].sum()
    here = frame.groupby(["match_id", "puuid", "cell"])[["observed", "expected"]].sum()
    out = seats[["match_id", *KEY]].drop_duplicates(["match_id", "puuid"])
    for cell in CELLS:
        totals = whole.xs(cell, level="cell") if cell in whole.index.get_level_values("cell") else whole.iloc[0:0].droplevel("cell")
        own = here.xs(cell, level="cell") if cell in here.index.get_level_values("cell") else here.iloc[0:0].droplevel("cell")
        joined = out.join(totals.rename(columns=lambda c: f"{c}_all"), on=KEY)
        joined = joined.join(own, on=["match_id", "puuid"]).fillna(0.0)
        others_seen = joined["observed_all"] - joined["observed"]
        others_due = joined["expected_all"] - joined["expected"]
        value = np.log((others_seen + prior) / (others_due + prior))
        out[f"{column}_{cell}"] = np.where(others_due > 0, value, 0.0)
    return out.drop(columns=["position"])


def build_tendencies(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    context = priority_context(settings)
    opportunities = with_priority(pd.read_parquet(settings.processed_dir / "opportunities.parquet"), context)
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"])
    out = seats[["match_id", "puuid"]].copy()
    everyone = seats[KEY].drop_duplicates().set_index(KEY)
    report = {"context": CONTEXT if len(context) else [], "cells": CELLS}
    shares = []
    for kind in KINDS:
        column = f"tend_{kind}"
        subset = opportunities[opportunities["kind"] == kind].merge(seats, on=["match_id", "puuid"], how="inner")
        if len(subset) < MIN_ROWS or subset["outcome"].nunique() < 2:
            for cell in CELLS:
                out[f"{column}_{cell}"] = 0.0
            continue
        design = pd.get_dummies(subset[[*COVARIATES, *CONTEXT, "role"]], columns=["role"], dtype=float)
        model = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000))]
        ).fit(design, subset["outcome"])
        expected = model.predict_proba(design)[:, 1]
        frame = pd.DataFrame(
            {
                "match_id": subset["match_id"].to_numpy(),
                "puuid": subset["puuid"].to_numpy(),
                "position": subset["position"].to_numpy(),
                "cell": cell_of(subset["top_priority"], subset["mid_priority"], subset["bot_priority"], subset["in_own_half"]),
                "observed": subset["outcome"].to_numpy(dtype=float),
                "expected": expected,
                "variance": expected * (1.0 - expected),
            }
        )
        pooled = frame.groupby(KEY)[["observed", "expected", "variance"]].sum()
        prior = _gamma_prior(pooled["observed"].to_numpy(), pooled["expected"].to_numpy(), pooled["variance"].to_numpy())
        expected = pooled["expected"].reindex(everyone.index, fill_value=0.0)
        shares.append((float(expected.mean()), (expected / (expected + prior)).rename("share").reset_index()))
        ratios = leave_one_out_ratio(frame, seats, prior, column)
        out = out.merge(ratios, on=["match_id", "puuid"], how="left")
        report[kind] = {
            "rows": int(len(subset)),
            "prior_strength": round(float(prior), 2),
            "cell_rows": frame["cell"].value_counts().to_dict(),
        }
    out = out[["match_id", "puuid", *TENDENCY_COLUMNS]].fillna(0.0)
    out.to_parquet(settings.processed_dir / TABLE, index=False)
    evidence = combine_shares(shares) if shares else everyone.assign(share=0.0).reset_index()
    evidence.to_parquet(settings.processed_dir / EVIDENCE, index=False)
    return {
        "rows": int(len(out)),
        "columns": len(TENDENCY_COLUMNS),
        "kinds": report,
        "evidence_share_median": round(float(evidence["share"].median()), 3),
    }


def load_tendencies(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *TENDENCY_COLUMNS])
    table = pd.read_parquet(path)
    if any(column not in table.columns for column in TENDENCY_COLUMNS):
        return pd.DataFrame(columns=["match_id", "puuid", *TENDENCY_COLUMNS])
    return table
