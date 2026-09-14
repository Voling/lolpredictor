import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import SVI, Trace_ELBO, autoguide

from ..config import Settings, get_settings

MIN_PAIR_GAMES = 5
STEPS = 20000
LEARNING_RATE = 0.02
SEED = 0
TEAM_SIZE = 5
TEAM_PAIRS = 10


def _gather(effect, index):
    return jnp.where(index >= 0, effect[jnp.clip(index, 0, None)], 0.0).sum(axis=1)


def model(player_a, player_b, pair_a, pair_b, n_players, n_pairs, win=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    player_scale = numpyro.sample("player_scale", dist.HalfNormal(0.5))
    pair_scale = numpyro.sample("pair_scale", dist.HalfNormal(0.5))
    with numpyro.plate("players", n_players):
        player = numpyro.sample("player", dist.Normal(0.0, player_scale))
    with numpyro.plate("pairs", n_pairs):
        pair_effect = numpyro.sample("pair_effect", dist.Normal(0.0, pair_scale))
    logit = (
        mean
        + player[player_a].sum(axis=1)
        - player[player_b].sum(axis=1)
        + _gather(pair_effect, pair_a)
        - _gather(pair_effect, pair_b)
    )
    with numpyro.plate("matches", logit.shape[0]):
        numpyro.sample("win", dist.Bernoulli(logits=logit), obs=win)


def design(pairs: pd.DataFrame, min_games: int = MIN_PAIR_GAMES) -> dict:
    frame = pairs.dropna(subset=["win"]).copy()
    low = np.where(frame.puuid_a < frame.puuid_b, frame.puuid_a, frame.puuid_b)
    high = np.where(frame.puuid_a < frame.puuid_b, frame.puuid_b, frame.puuid_a)
    frame["puuid_a"], frame["puuid_b"] = low, high
    frame["pair_key"] = frame.puuid_a + "|" + frame.puuid_b
    counts = frame.pair_key.value_counts()
    repeat = pd.Index(sorted(counts[counts >= min_games].index))
    frame["pair_slot"] = repeat.get_indexer(frame.pair_key)
    frame = frame.sort_values(["match_id", "team_id", "pair_key"]).reset_index(drop=True)

    sizes = frame.groupby(["match_id", "team_id"], sort=True).size()
    full = sizes[sizes == TEAM_PAIRS].index
    frame = frame.set_index(["match_id", "team_id"]).loc[full].reset_index()
    frame = frame.sort_values(["match_id", "team_id", "pair_key"]).reset_index(drop=True)

    sides = frame[["match_id", "team_id", "win"]].drop_duplicates(
        subset=["match_id", "team_id"]
    )
    both = sides.match_id.value_counts()
    keep = set(both[both == 2].index)
    mask = frame.match_id.isin(keep)
    frame, sides = frame[mask].reset_index(drop=True), sides[sides.match_id.isin(keep)]

    roster = (
        pd.concat(
            [
                frame[["match_id", "team_id", "puuid_a"]].rename(columns={"puuid_a": "puuid"}),
                frame[["match_id", "team_id", "puuid_b"]].rename(columns={"puuid_b": "puuid"}),
            ]
        )
        .drop_duplicates()
        .sort_values(["match_id", "team_id", "puuid"])
        .reset_index(drop=True)
    )
    if len(roster) != len(sides) * TEAM_SIZE:
        raise ValueError(f"roster is {len(roster)} rows, expected {len(sides) * TEAM_SIZE}")

    players = pd.Index(sorted(roster.puuid.unique()))
    seats = players.get_indexer(roster.puuid).reshape(-1, TEAM_SIZE)
    slots = frame.pair_slot.to_numpy().reshape(-1, TEAM_PAIRS)
    ordered = sides.sort_values(["match_id", "team_id"])
    outcome = ordered.win.to_numpy().reshape(-1, 2)
    return {
        "matches": ordered.match_id.to_numpy()[0::2],
        "player_a": seats[0::2],
        "player_b": seats[1::2],
        "pair_a": slots[0::2],
        "pair_b": slots[1::2],
        "win": outcome[:, 0].astype(int),
        "n_players": len(players),
        "n_pairs": int(slots.max()) + 1,
        "repeat_rows": int((slots >= 0).sum()),
        "players": players,
        "repeat": repeat,
    }


def fit_variance(
    pairs: pd.DataFrame,
    settings: Settings | None = None,
    steps: int = STEPS,
    min_games: int = MIN_PAIR_GAMES,
    seed: int = SEED,
    report_path: str = "variance_report.json",
    shuffle: int | None = None,
) -> dict:
    settings = settings or get_settings()
    built = design(pairs, min_games)
    if len(built["win"]) < 2000 or built["n_pairs"] < 2:
        return {}
    win = built["win"]
    if shuffle is not None:
        win = np.random.default_rng(shuffle).permutation(win)
    arguments = tuple(
        jnp.asarray(built[name]) for name in ("player_a", "player_b", "pair_a", "pair_b")
    ) + (built["n_players"], built["n_pairs"])
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(LEARNING_RATE), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed), steps, *arguments,
        win=jnp.asarray(win), progress_bar=False,
    )
    quantiles = guide.quantiles(result.params, [0.025, 0.5, 0.975])
    report = {
        "matches": int(len(built["win"])),
        "players": built["n_players"],
        "repeat_pairs": built["n_pairs"],
        "repeat_rows": built["repeat_rows"],
        "min_pair_games": min_games,
        "shuffled": shuffle is not None,
        "steps": steps,
        "final_loss": round(float(result.losses[-1]), 3),
        "loss_drift": round(
            float(np.mean(result.losses[-800:-400]) - np.mean(result.losses[-400:])), 4
        ),
        "player_scale": [round(float(v), 5) for v in quantiles["player_scale"]],
        "pair_scale": [round(float(v), 5) for v in quantiles["pair_scale"]],
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_dir / report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
