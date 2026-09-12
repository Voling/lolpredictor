import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .policy import ACTIONS, STATES
from .valuesurface import load_value_surface

COMPLEMENT_COLUMNS = ["policy_joint", "policy_solo", "policy_distance"]
MIN_ROWS = 24


def player_priors(policy: pd.DataFrame, smoothing: float = 8.0) -> pd.DataFrame:
    counts = policy.groupby(["puuid", "state"]).size().rename("n")
    acts = policy.groupby(["puuid", "state", "action"]).size().rename("k")
    base = policy.groupby(["state", "action"]).size() / policy.groupby("state").size()
    frame = acts.to_frame().join(counts, on=["puuid", "state"])
    frame["prior"] = base.reindex(frame.index.droplevel(0)).to_numpy()
    frame["p"] = (frame["k"] + smoothing * frame["prior"]) / (frame["n"] + smoothing)
    seen = policy.groupby("puuid").size()
    keep = set(seen[seen >= MIN_ROWS].index)
    return frame[frame.index.get_level_values(0).isin(keep)]["p"]


def _surface_lookup(surface: pd.DataFrame) -> dict:
    if surface.empty:
        return {}
    keys = zip(
        surface["role_a"], surface["role_b"], surface["state_a"],
        surface["action_a"], surface["action_b"], surface["value"],
    )
    return {(a, b, s, x, y): v for a, b, s, x, y, v in keys}


def complement_table(
    policy: pd.DataFrame, pairs: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    surface = _surface_lookup(load_value_surface(settings))
    if not surface:
        return pd.DataFrame(columns=["puuid_a", "puuid_b", *COMPLEMENT_COLUMNS])
    priors = player_priors(policy)
    mix = (policy.groupby("state").size() / max(len(policy), 1)).to_dict()
    base = (policy.groupby(["state", "action"]).size() / policy.groupby("state").size()).to_dict()
    roles = policy.groupby("puuid")["role"].agg(lambda values: values.mode().iat[0])
    lookup = priors.to_dict()
    known = set(roles.index)

    def expected(left: str, right: str, average_partner: bool) -> float:
        role_a, role_b = roles.get(left), roles.get(right)
        if role_a is None or role_b is None or role_a == role_b:
            return 0.0
        first, second = (left, right) if role_a < role_b else (right, left)
        low, high = sorted((role_a, role_b))
        total = 0.0
        for state in STATES:
            weight = mix.get(state, 0.0)
            if weight <= 0.0:
                continue
            for action_a in ACTIONS:
                pa = lookup.get((first, state, action_a), base.get((state, action_a), 0.0))
                if pa < 1e-4:
                    continue
                for action_b in ACTIONS:
                    pb = (
                        base.get((state, action_b), 0.0)
                        if average_partner
                        else lookup.get((second, state, action_b), base.get((state, action_b), 0.0))
                    )
                    if pb < 1e-4:
                        continue
                    total += weight * pa * pb * surface.get(
                        (low, high, state, action_a, action_b), 0.0
                    )
        return total

    def distance(left: str, right: str) -> float:
        cells = [(s, a) for s in STATES for a in ACTIONS]
        vector_a = np.array([lookup.get((left, s, a), base.get((s, a), 0.0)) for s, a in cells])
        vector_b = np.array([lookup.get((right, s, a), base.get((s, a), 0.0)) for s, a in cells])
        norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        return 1.0 - float(vector_a @ vector_b / norm) if norm > 0 else 0.0

    wanted = pairs[["puuid_a", "puuid_b"]].drop_duplicates()
    wanted = wanted[wanted.puuid_a.isin(known) & wanted.puuid_b.isin(known)]
    rows = []
    for left, right in wanted.itertuples(index=False):
        joint = expected(left, right, average_partner=False)
        solo = expected(left, right, average_partner=True) + expected(
            right, left, average_partner=True
        )
        rows.append(
            {
                "puuid_a": left,
                "puuid_b": right,
                "policy_joint": joint,
                "policy_solo": solo,
                "policy_distance": distance(left, right),
            }
        )
    frame = pd.DataFrame(rows, columns=["puuid_a", "puuid_b", *COMPLEMENT_COLUMNS])
    if not frame.empty:
        frame.to_parquet(settings.processed_dir / "complement.parquet", index=False)
    return frame


def load_complement(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / "complement.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["puuid_a", "puuid_b", *COMPLEMENT_COLUMNS])
    return pd.read_parquet(path)
