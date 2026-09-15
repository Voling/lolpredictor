import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings, get_settings
from .propensity import COVARIATES, KINDS, _gamma_prior

TENDENCY_COLUMNS = [f"tend_{kind}" for kind in KINDS]
TABLE = "tendency.parquet"
MIN_ROWS = 200


def build_tendencies(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    opportunities = pd.read_parquet(settings.processed_dir / "opportunities.parquet")
    seats = pd.read_parquet(
        settings.processed_dir / "participations.parquet", columns=["match_id", "puuid"]
    )
    out = seats.copy()
    report = {}
    for kind in KINDS:
        column = f"tend_{kind}"
        subset = opportunities[opportunities["kind"] == kind]
        if len(subset) < MIN_ROWS or subset["outcome"].nunique() < 2:
            out[column] = np.nan
            continue
        design = pd.get_dummies(subset[[*COVARIATES, "role"]], columns=["role"], dtype=float)
        model = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000))]
        ).fit(design, subset["outcome"])
        expected = model.predict_proba(design)[:, 1]
        frame = pd.DataFrame(
            {
                "match_id": subset["match_id"].to_numpy(),
                "puuid": subset["puuid"].to_numpy(),
                "observed": subset["outcome"].to_numpy(dtype=float),
                "expected": expected,
                "variance": expected * (1.0 - expected),
            }
        )
        whole = frame.groupby("puuid")[["observed", "expected", "variance"]].sum()
        prior = _gamma_prior(
            whole["observed"].to_numpy(), whole["expected"].to_numpy(), whole["variance"].to_numpy()
        )
        here = frame.groupby(["match_id", "puuid"])[["observed", "expected"]].sum()
        joined = seats.join(whole[["observed", "expected"]], on="puuid").rename(
            columns={"observed": "observed_all", "expected": "expected_all"}
        )
        joined = joined.join(here, on=["match_id", "puuid"]).fillna(
            {"observed": 0.0, "expected": 0.0, "observed_all": 0.0, "expected_all": 0.0}
        )
        others_seen = joined["observed_all"] - joined["observed"]
        others_due = joined["expected_all"] - joined["expected"]
        value = np.log((others_seen + prior) / (others_due + prior))
        value[others_due <= 0] = np.nan
        out[column] = value.to_numpy()
        report[kind] = {
            "rows": int(len(subset)),
            "prior_strength": round(float(prior), 2),
            "coverage": round(float(out[column].notna().mean()), 4),
        }
    for column in TENDENCY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out[["match_id", "puuid", *TENDENCY_COLUMNS]]
    out.to_parquet(settings.processed_dir / TABLE, index=False)
    return {"rows": int(len(out)), "kinds": report}


def load_tendencies(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *TENDENCY_COLUMNS])
    return pd.read_parquet(path)
