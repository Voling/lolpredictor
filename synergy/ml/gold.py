import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import SVI, Trace_ELBO, autoguide

from ..config import Settings, get_settings
from ..features.advantage import STATE_COLUMNS, load_evaluation
from ..ingest.store import Store

MIN_PAIR_GAMES = 5
LEARNING_RATE = 0.02
SEED = 0
STEPS = 20000
MINUTE = 15
TEAM_SIZE = 5
TEAM_PAIRS = 10
OBJECTIVES = {"DRAGON": "dragon_diff", "HORDE": "grub_diff"}


def _gather(effect, index):
    return jnp.where(index >= 0, effect[jnp.clip(index, 0, None)], 0.0).sum(axis=1)


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


def objective_prices(settings: Settings) -> dict:
    model = load_evaluation(settings)
    if not model:
        return {}
    weights = model["weights"]
    effective = {c: weights[c] + weights[f"{c}_x_minute"] for c in STATE_COLUMNS}
    gold = effective["gold_diff"]
    if not gold:
        return {}
    return {
        monster: effective[column] / gold * 1000.0
        for monster, column in OBJECTIVES.items()
        if column in effective
    }


def team_objectives(settings: Settings, minute: int = MINUTE) -> pd.DataFrame:
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute(
                "SELECT match_id, killer_team_id AS team_id, monster_type,"
                " COUNT(*) AS taken FROM events"
                " WHERE type = 'ELITE_MONSTER_KILL' AND minute <= %s"
                " AND monster_type = ANY(%s) AND killer_team_id IS NOT NULL"
                " GROUP BY match_id, killer_team_id, monster_type",
                (minute, list(OBJECTIVES)),
            )
            rows = cursor.fetchall()
    finally:
        store.close()
    if not rows:
        return pd.DataFrame(columns=["match_id", "team_id", "monster_type", "taken"])
    return pd.DataFrame(rows)


def team_advantage(settings: Settings, minute: int = MINUTE) -> pd.Series:
    gold = team_gold(settings, minute)
    prices = objective_prices(settings)
    taken = team_objectives(settings, minute)
    if not prices:
        raise ValueError("no evaluation weights, cannot price objectives")
    if taken.empty:
        raise ValueError("no objective events found, cannot price objectives")
    taken["value"] = taken.monster_type.map(prices) * taken.taken
    wide = taken.pivot_table(
        index="match_id", columns="team_id", values="value", aggfunc="sum", fill_value=0.0
    )
    for team in (100, 200):
        if team not in wide.columns:
            wide[team] = 0.0
    edge = (wide[100] - wide[200]).astype(float)
    return gold.add(edge.reindex(gold.index).fillna(0.0))


def team_gold(settings: Settings, minute: int = MINUTE) -> pd.Series:
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute(
                "SELECT f.match_id, p.team_id, SUM(f.total_gold) AS gold, COUNT(*) AS seats"
                " FROM frames f JOIN participations p"
                " ON p.match_id = f.match_id AND p.puuid = f.puuid"
                " WHERE f.minute = %s GROUP BY f.match_id, p.team_id",
                (minute,),
            )
            rows = cursor.fetchall()
    finally:
        store.close()
    frame = pd.DataFrame(rows)
    frame = frame[frame.seats == TEAM_SIZE]
    wide = frame.pivot(index="match_id", columns="team_id", values="gold").dropna()
    return (wide[100] - wide[200]).astype(float)


def model(player_a, player_b, pair_a, pair_b, n_players, n_pairs, gold=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    player_scale = numpyro.sample("player_scale", dist.HalfNormal(1.0))
    pair_scale = numpyro.sample("pair_scale", dist.HalfNormal(1.0))
    residual = numpyro.sample("residual", dist.HalfNormal(1.0))
    with numpyro.plate("players", n_players):
        player = numpyro.sample("player", dist.Normal(0.0, player_scale))
    with numpyro.plate("pairs", n_pairs):
        effect = numpyro.sample("pair_effect", dist.Normal(0.0, pair_scale))
    centre = (
        mean
        + player[player_a].sum(axis=1)
        - player[player_b].sum(axis=1)
        + _gather(effect, pair_a)
        - _gather(effect, pair_b)
    )
    with numpyro.plate("matches", centre.shape[0]):
        numpyro.sample("gold", dist.Normal(centre, residual), obs=gold)


def fit_gold(
    pairs: pd.DataFrame,
    settings: Settings | None = None,
    steps: int = STEPS,
    min_games: int = MIN_PAIR_GAMES,
    seed: int = SEED,
    shuffle: int | None = None,
    objectives: bool = True,
    outcome: pd.Series | None = None,
    report_path: str = "gold_report.json",
) -> dict:
    settings = settings or get_settings()
    built = design(pairs, min_games)
    gold = outcome if outcome is not None else (
        team_advantage(settings) if objectives else team_gold(settings)
    )
    index = pd.Index(built["matches"])
    aligned = gold.reindex(index)
    keep = aligned.notna().to_numpy()
    if keep.sum() < 2000 or built["n_pairs"] < 2:
        return {}
    target = aligned.to_numpy()[keep]
    scale = float(target.std())
    target = (target - target.mean()) / scale
    if shuffle is not None:
        target = np.random.default_rng(shuffle).permutation(target)
    arguments = tuple(
        jnp.asarray(built[name][keep])
        for name in ("player_a", "player_b", "pair_a", "pair_b")
    ) + (built["n_players"], built["n_pairs"])

    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(LEARNING_RATE), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed), steps, *arguments,
        gold=jnp.asarray(target), progress_bar=False,
    )
    quantiles = guide.quantiles(result.params, [0.025, 0.5, 0.975])
    report = {
        "matches": int(keep.sum()),
        "players": built["n_players"],
        "repeat_pairs": built["n_pairs"],
        "repeat_rows": built["repeat_rows"],
        "min_pair_games": min_games,
        "shuffled": shuffle is not None,
        "gold_sd": round(scale, 1),
        "objectives": objectives,
        "steps": steps,
        "final_loss": round(float(result.losses[-1]), 3),
        "loss_drift": round(
            float(np.mean(result.losses[-800:-400]) - np.mean(result.losses[-400:])), 4
        ),
        "player_scale": [round(float(v), 5) for v in quantiles["player_scale"]],
        "pair_scale": [round(float(v), 5) for v in quantiles["pair_scale"]],
        "residual": [round(float(v), 5) for v in quantiles["residual"]],
    }
    pair = float(quantiles["pair_scale"][1])
    report["gold"] = {
        "per_pair_sd": round(pair * scale, 1),
        "team_sum_sd": round(pair * scale * (TEAM_PAIRS ** 0.5), 1),
        "residual_sd": round(float(quantiles["residual"][1]) * scale, 1),
        "share_of_variance": round(
            (pair ** 2 * TEAM_PAIRS * 2)
            / (pair ** 2 * TEAM_PAIRS * 2 + float(quantiles["residual"][1]) ** 2),
            5,
        ),
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_dir / report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
