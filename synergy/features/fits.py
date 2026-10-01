import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings, get_settings
from .cells import CellFit, SeatIndex, buffer, combine_shares, dense_counts, fit_cells, moment_kappa
from .positions import KEY
from .propensity import COVARIATES, KINDS, _gamma_prior

PRIORITY_FIT = "cells_prio.npz"
REACTION_FITS = ("cells_rsp.npz", "cells_obj.npz", "cells_ward.npz", "cells_jgl.npz", "cells_jgl_sides.npz")
TENDENCY_FIT = "tendency_fit.pkl"
FIT_FILES = (PRIORITY_FIT, *REACTION_FITS, TENDENCY_FIT)


def save_cells(path: Path, fit: CellFit, positions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        prefix=np.array(fit.prefix),
        situations=np.array(fit.situations),
        outcomes=np.array(fit.outcomes),
        worlds=fit.worlds,
        kappa=fit.kappa,
        unit=np.array(fit.unit),
        positions=np.array(positions),
        typical=fit.totals.sum(axis=2).mean(axis=0),
    )


def load_cells(path: Path) -> dict:
    with np.load(path) as data:
        return {
            "prefix": str(data["prefix"]),
            "situations": [str(name) for name in data["situations"]],
            "outcomes": [str(name) for name in data["outcomes"]],
            "worlds": data["worlds"],
            "kappa": data["kappa"],
            "unit": float(data["unit"]),
            "positions": [str(name) for name in data["positions"]],
            "typical": data["typical"],
        }


def cell_columns(saved: dict) -> list[str]:
    return [f"{saved['prefix']}_{situation}_{outcome}" for situation in saved["situations"] for outcome in saved["outcomes"]]


def apply_cells(saved: dict, counts: np.ndarray, index: SeatIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    situations, outcomes = saved["situations"], saved["outcomes"]
    shape = (len(index), len(situations), len(outcomes))
    values = np.asarray(counts, dtype=float).reshape(shape) / saved["unit"]
    onehot = sparse.csr_matrix((np.ones(len(index)), (index.who_code, np.arange(len(index)))), shape=(len(index.who), len(index)))
    totals = np.asarray(onehot @ values.reshape(len(index), -1)).reshape(len(index.who), len(situations), len(outcomes))
    code = pd.Index(saved["positions"]).get_indexer(index.positions)[index.seat_position]
    known = code >= 0
    others = totals[index.who_code] - values
    exposure = others.sum(axis=2, keepdims=True)
    world = saved["worlds"][:, np.where(known, code, 0)].transpose(1, 0, 2)
    kappa = saved["kappa"][None, :, None]
    cells = ((world * kappa + others) / (exposure + kappa)).reshape(len(index), -1)
    cells[~known] = np.nan
    frame = pd.DataFrame(cells, columns=cell_columns(saved))
    frame.insert(0, "puuid", index.puuid)
    frame.insert(0, "match_id", index.match_id)
    who_exposure = totals.sum(axis=2)
    rounded = np.round(saved["kappa"], 2)
    weight = who_exposure / (who_exposure + rounded[None, :])
    typical = saved["typical"]
    share = (weight * typical[None, :]).sum(axis=1) / max(float(typical.sum()), 1e-9)
    evidence = pd.DataFrame({"puuid": index.who.get_level_values("puuid"), "position": index.who.get_level_values("position"), "share": share})
    return frame, evidence


def _with_context(rows: pd.DataFrame, context: pd.DataFrame, fill: dict[str, float]) -> pd.DataFrame:
    from .tendency import CONTEXT

    if context.empty:
        return rows.assign(**{column: fill.get(column, 0.0) for column in CONTEXT})
    lagged = context.assign(minute=context["minute"] + 1)
    joined = rows.merge(lagged, on=["match_id", "puuid", "minute"], how="left")
    for column in CONTEXT:
        joined[column] = joined[column].fillna(fill.get(column, 0.0))
    return joined


def _design(subset: pd.DataFrame) -> pd.DataFrame:
    from .tendency import CONTEXT

    return pd.get_dummies(subset[[*COVARIATES, *CONTEXT, "role"]], columns=["role"], dtype=float)


def _frame(subset: pd.DataFrame, expected: np.ndarray) -> pd.DataFrame:
    from .tendency import cell_of

    return pd.DataFrame(
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


def fit_kind(subset: pd.DataFrame, everyone: pd.Index, scale: float, fill: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    design = _design(subset)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000))]).fit(design, subset["outcome"])
    frame = _frame(subset, model.predict_proba(design)[:, 1])
    pooled = frame.groupby(KEY)[["observed", "expected", "variance"]].sum()
    prior = scale * _gamma_prior(pooled["observed"].to_numpy(), pooled["expected"].to_numpy(), pooled["variance"].to_numpy())
    weight = float(pooled["expected"].reindex(everyone, fill_value=0.0).mean())
    return {"model": model, "columns": list(design.columns), "prior": float(prior), "weight": weight, "fill": dict(fill)}, frame


def fit_tendencies(settings: Settings | None = None) -> dict:
    from .priority import priority_context
    from .tendency import CONTEXT, MIN_ROWS

    settings = settings or get_settings()
    context = priority_context(settings)
    fill = {column: float(context[column].mean()) for column in CONTEXT} if len(context) else {column: 0.0 for column in CONTEXT}
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"])
    everyone = seats[KEY].drop_duplicates().set_index(KEY).index
    fits = {}
    for kind in KINDS:
        rows = pd.read_parquet(settings.processed_dir / "opportunities.parquet", filters=[("kind", "==", kind)])
        subset = _with_context(rows, context, fill).merge(seats, on=["match_id", "puuid"], how="inner")
        del rows
        if len(subset) < MIN_ROWS or subset["outcome"].nunique() < 2:
            fits[kind] = None
            continue
        fits[kind], _ = fit_kind(subset, everyone, settings.cell_prior_scale, fill)
    return fits


def save_tendencies(path: Path, fits: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(fits, handle)


def load_tendencies(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def apply_tendencies(fits: dict, opportunities: pd.DataFrame, context: pd.DataFrame, seats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .tendency import TENDENCY_COLUMNS, leave_one_out_ratio

    out = seats[["match_id", "puuid"]].drop_duplicates().copy()
    everyone = seats[KEY].drop_duplicates().set_index(KEY)
    shares = []
    for kind in KINDS:
        fit = fits.get(kind)
        rows = opportunities[opportunities["kind"] == kind]
        if fit is None or rows.empty:
            continue
        subset = _with_context(rows, context, fit["fill"]).merge(seats, on=["match_id", "puuid"], how="inner")
        if subset.empty:
            continue
        design = _design(subset).reindex(columns=fit["columns"], fill_value=0.0)
        frame = _frame(subset, fit["model"].predict_proba(design)[:, 1])
        expected = frame.groupby(KEY)["expected"].sum().reindex(everyone.index, fill_value=0.0)
        shares.append((fit["weight"], (expected / (expected + fit["prior"])).rename("share").reset_index()))
        out = out.merge(leave_one_out_ratio(frame, seats, fit["prior"], f"tend_{kind}"), on=["match_id", "puuid"], how="left")
    for column in TENDENCY_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    out = out[["match_id", "puuid", *TENDENCY_COLUMNS]].fillna(0.0)
    evidence = combine_shares(shares) if shares else everyone.assign(share=0.0).reset_index()
    return out, evidence


def fit_priority_from_buffer(settings: Settings) -> dict | None:
    from ..ingest.store import Store
    from .priority import COUNTS, OUTCOMES, SITUATIONS, TICK_UNIT

    path = settings.processed_dir / "buffers" / COUNTS
    if not path.exists():
        return None
    store = Store(settings)
    try:
        match_ids = set(store.corpus_ids())
    finally:
        store.close()
    seats = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", "puuid", "position"])
    index = SeatIndex(seats[seats.match_id.isin(match_ids)])
    counts = np.load(path, mmap_mode="r")
    if counts.shape[0] != len(index):
        raise RuntimeError(f"the priority counts buffer has {counts.shape[0]:,} seats but the corpus has {len(index):,}; rebuild the blocks first")
    kappas = [moment_kappa(counts, index, step) for step in range(len(SITUATIONS))]
    fit = fit_cells(counts, index, SITUATIONS, OUTCOMES, "prio", scale=settings.cell_prior_scale, unit=TICK_UNIT, kappa=kappas)
    save_cells(settings.model_dir / PRIORITY_FIT, fit, index.positions)
    return {"seats": len(index), "kappa": [round(float(k), 2) for k in fit.kappa]}


def reaction_sources(settings: Settings) -> list[tuple[str, object, list[str], list[str]]]:
    from .reaction import (
        OBJECTIVE_READ,
        OBJECTIVE_RESPONSES,
        OBJECTIVE_SITUATIONS,
        OPENING_OUTCOMES,
        OPENING_SITUATIONS,
        RESPONSE_SITUATIONS,
        RESPONSES,
        SIDES,
        WARD_SITUATIONS,
        WARD_ZONES,
        _pieces,
        _responses,
        jungle_counts,
        objective_counts,
        response_counts,
        ward_counts,
    )

    processed = settings.processed_dir
    openings = jungle_counts(pd.read_parquet(processed / "jungle_openings.parquet"))
    return [
        ("rsp", lambda: (response_counts(piece) for piece in _responses(settings)), RESPONSE_SITUATIONS, list(RESPONSES)),
        ("obj", lambda: (objective_counts(piece) for piece in _pieces(processed / "objectives.parquet", OBJECTIVE_READ)), OBJECTIVE_SITUATIONS, list(OBJECTIVE_RESPONSES)),
        ("ward", lambda: (ward_counts(piece) for piece in _pieces(processed / "wards.parquet", ["match_id", "puuid", "minute", "zone"])), WARD_SITUATIONS, list(WARD_ZONES)),
        ("jgl", lambda: [openings[openings["situation"] != "sides"]], OPENING_SITUATIONS, OPENING_OUTCOMES),
        ("jgl", lambda: [openings[openings["situation"] == "sides"]], ["sides"], list(SIDES)),
    ]


def fit_reaction_from_tables(settings: Settings) -> dict:
    processed = settings.processed_dir
    index = SeatIndex(pd.read_parquet(processed / "participations.parquet", columns=["match_id", "puuid", "position"]))
    report = {}
    for number, (prefix, load, situations, outcomes) in enumerate(reaction_sources(settings)):
        path = processed / "buffers" / f"reaction_fit.{number}.npy"
        counts = buffer(path, index, situations, outcomes)
        for frame in load():
            dense_counts(frame, index, situations, outcomes, out=counts)
        counts.flush()
        fit = fit_cells(counts, index, situations, outcomes, prefix, scale=settings.cell_prior_scale)
        save_cells(settings.model_dir / REACTION_FITS[number], fit, index.positions)
        report[REACTION_FITS[number]] = {"kappa": [round(float(k), 2) for k in fit.kappa]}
        del counts
        path.unlink(missing_ok=True)
    return report


def write_fits(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    report = {"priority": fit_priority_from_buffer(settings)}
    report["reaction"] = fit_reaction_from_tables(settings)
    fits = fit_tendencies(settings)
    save_tendencies(settings.model_dir / TENDENCY_FIT, fits)
    report["tendency"] = {kind: None if fit is None else {"prior": round(fit["prior"], 2), "weight": round(fit["weight"], 3)} for kind, fit in fits.items()}
    return report
