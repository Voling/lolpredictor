import pandas as pd

from ..config import Settings, get_settings
from .propensity import _gamma_prior

RESPONSES = ["converged", "present", "nearby"]
MIN_TRIGGERS = 5
TABLE = "hinge.parquet"
HINGE_COLUMNS = [f"hinge_{name}" for name in RESPONSES] + ["hinge_triggers"]


def _pairs(responses: pd.DataFrame) -> pd.DataFrame:
    actors = responses[responses.is_actor == 1][["match_id", "trigger_id", "puuid"]].rename(
        columns={"puuid": "actor"}
    )
    replies = responses[(responses.ours == 1) & (responses.is_actor == 0)][
        ["match_id", "trigger_id", "puuid", *RESPONSES]
    ]
    return replies.merge(actors, on=["match_id", "trigger_id"], how="inner")


def build_hinge(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    responses = pd.read_parquet(
        settings.processed_dir / "event_responses.parquet",
        columns=["match_id", "trigger_id", "puuid", "ours", "is_actor", *RESPONSES],
    )
    joined = _pairs(responses)
    baseline = joined.groupby("puuid")[RESPONSES].mean()
    together = joined.groupby(["puuid", "actor"])
    counts = together.size().rename("hinge_triggers")
    observed = together[RESPONSES].sum()
    expected = baseline.reindex(observed.index.get_level_values("puuid")).to_numpy() * counts.to_numpy()[:, None]
    variance = expected * (1.0 - baseline.reindex(observed.index.get_level_values("puuid")).to_numpy())

    out = pd.DataFrame(index=observed.index)
    strengths = {}
    for column_index, name in enumerate(RESPONSES):
        seen, hoped, spread = (
            observed[name].to_numpy(dtype=float),
            expected[:, column_index],
            variance[:, column_index],
        )
        strength = _gamma_prior(seen, hoped, spread)
        strengths[name] = round(float(strength), 3)
        out[f"hinge_{name}"] = (seen + strength) / (hoped + strength)
    out["hinge_triggers"] = counts
    out = out[out.hinge_triggers >= MIN_TRIGGERS].reset_index()
    out = out.rename(columns={"puuid": "responder"})
    out.to_parquet(settings.processed_dir / TABLE, index=False)
    return {
        "responder_actor_pairs": int(len(out)),
        "triggers_joined": int(len(joined)),
        "prior_strength": strengths,
        "baseline_rates": {name: round(float(baseline[name].mean()), 4) for name in RESPONSES},
    }


def load_hinge(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / TABLE
    if not path.exists():
        return pd.DataFrame(columns=["responder", "actor", *HINGE_COLUMNS])
    return pd.read_parquet(path)
