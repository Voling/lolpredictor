import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .policy import ACTIONS, POLICY_COLUMNS, STATES
from .valuesurface import design, load_value_model, load_value_surface

COMPLEMENT_COLUMNS = ["policy_joint", "policy_solo", "policy_distance"]
MIN_ROWS = 24
FLOOR = 1e-3


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


def _role_slots(roles: list[str]) -> dict:
    pairs = [(low, high) for index, low in enumerate(roles) for high in roles[index + 1 :]]
    return {pair: slot for slot, pair in enumerate(pairs)}


def value_grid(booster, surface: pd.DataFrame, context: pd.DataFrame, slots: dict) -> np.ndarray:
    grid = np.zeros((len(slots), len(STATES), len(ACTIONS), len(ACTIONS)))
    if booster is not None:
        rows, cells = [], []
        for (low, high), slot in slots.items():
            for si, state in enumerate(STATES):
                if state not in context.index:
                    continue
                ctx = context.loc[state]
                for ai, action_a in enumerate(ACTIONS):
                    for bi, action_b in enumerate(ACTIONS):
                        row = {f"{c}_a": ctx[c] for c in POLICY_COLUMNS}
                        row.update({f"{c}_b": ctx[c] for c in POLICY_COLUMNS})
                        row.update(
                            {
                                "role_a": low,
                                "role_b": high,
                                "action_a": action_a,
                                "action_b": action_b,
                            }
                        )
                        rows.append(row)
                        cells.append((slot, si, ai, bi))
        if rows:
            predicted = booster.predict(design(pd.DataFrame(rows)))
            for (slot, si, ai, bi), value in zip(cells, predicted):
                grid[slot, si, ai, bi] = value
        return grid
    if surface.empty:
        return grid
    states = {state: si for si, state in enumerate(STATES)}
    actions = {action: ai for ai, action in enumerate(ACTIONS)}
    for row in surface.itertuples(index=False):
        slot = slots.get((row.role_a, row.role_b))
        si = states.get(row.state_a)
        ai, bi = actions.get(row.action_a), actions.get(row.action_b)
        if slot is None or si is None or ai is None or bi is None:
            continue
        grid[slot, si, ai, bi] = row.value
    return grid


def _base_array(base: dict) -> np.ndarray:
    table = np.zeros((len(STATES), len(ACTIONS)))
    for si, state in enumerate(STATES):
        for ai, action in enumerate(ACTIONS):
            table[si, ai] = base.get((state, action), 0.0)
    return table


def _prior_array(priors: dict, base: dict, players: pd.Index) -> np.ndarray:
    seats = {puuid: seat for seat, puuid in enumerate(players)}
    states = {state: si for si, state in enumerate(STATES)}
    actions = {action: ai for ai, action in enumerate(ACTIONS)}
    table = np.repeat(_base_array(base)[None], len(players), axis=0)
    for (puuid, state, action), value in priors.items():
        seat, si, ai = seats.get(puuid), states.get(state), actions.get(action)
        if seat is None or si is None or ai is None:
            continue
        table[seat, si, ai] = value
    return table


def _ratio(total: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.divide(total, mass, out=np.zeros_like(total), where=mass > 0)


def complement_table(
    policy: pd.DataFrame, pairs: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    booster = load_value_model(settings)
    surface = load_value_surface(settings)
    if booster is None and surface.empty:
        return pd.DataFrame(columns=["puuid_a", "puuid_b", *COMPLEMENT_COLUMNS])

    priors = player_priors(policy).to_dict()
    mix = policy.groupby("state").size() / max(len(policy), 1)
    share = np.array([float(mix.get(state, 0.0)) for state in STATES])
    base = (policy.groupby(["state", "action"]).size() / policy.groupby("state").size()).to_dict()
    roles = policy.groupby("puuid")["role"].agg(lambda values: values.mode().iat[0])
    context = policy.groupby("state")[POLICY_COLUMNS].mean()

    slots = _role_slots(sorted(roles.unique()))
    grid = value_grid(booster, surface, context, slots)
    players = pd.Index(sorted(roles.index))
    roles = roles.reindex(players)
    raw = _prior_array(priors, base, players)
    masked = np.where(raw < FLOOR, 0.0, raw)
    generic = np.where(_base_array(base) < FLOOR, 0.0, _base_array(base))

    wanted = pairs[["puuid_a", "puuid_b"]].drop_duplicates()
    seat_a = players.get_indexer(wanted.puuid_a)
    seat_b = players.get_indexer(wanted.puuid_b)
    keep = (seat_a >= 0) & (seat_b >= 0)
    wanted, seat_a, seat_b = wanted[keep], seat_a[keep], seat_b[keep]
    role_a, role_b = roles.to_numpy()[seat_a], roles.to_numpy()[seat_b]
    slot = np.array([slots.get((min(x, y), max(x, y)), -1) for x, y in zip(role_a, role_b)])
    low_seat = np.where(role_a < role_b, seat_a, seat_b)
    high_seat = np.where(role_a < role_b, seat_b, seat_a)

    joint, solo = np.zeros(len(wanted)), np.zeros(len(wanted))
    for index in range(len(slots)):
        rows = np.flatnonzero(slot == index)
        if not len(rows):
            continue
        table = grid[index]
        low, high = masked[low_seat[rows]], masked[high_seat[rows]]
        joint[rows] = _ratio(
            np.einsum("nsa,sab,nsb,s->n", low, table, high, share),
            np.einsum("nsa,nsb,s->n", low, high, share),
        )
        solo[rows] = _ratio(
            np.einsum("nsa,sab,sb,s->n", low, table, generic, share),
            np.einsum("nsa,sb,s->n", low, generic, share),
        ) + _ratio(
            np.einsum("sa,sab,nsb,s->n", generic, table, high, share),
            np.einsum("sa,nsb,s->n", generic, high, share),
        )

    flat = raw.reshape(len(players), -1)
    norms = np.linalg.norm(flat, axis=1)
    scale = norms[seat_a] * norms[seat_b]
    distance = 1.0 - _ratio(np.einsum("nk,nk->n", flat[seat_a], flat[seat_b]), scale)

    same = role_a == role_b
    frame = pd.DataFrame(
        {
            "puuid_a": wanted.puuid_a.to_numpy(),
            "puuid_b": wanted.puuid_b.to_numpy(),
            "policy_joint": np.where(same, 0.0, joint),
            "policy_solo": np.where(same, 0.0, solo),
            "policy_distance": distance,
        }
    )
    if not frame.empty:
        frame.to_parquet(settings.processed_dir / "complement.parquet", index=False)
    return frame


def load_complement(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / "complement.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["puuid_a", "puuid_b", *COMPLEMENT_COLUMNS])
    return pd.read_parquet(path)
