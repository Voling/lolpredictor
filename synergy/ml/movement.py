import json
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary as mcmc_summary
from numpyro.infer import MCMC, NUTS
from psycopg.rows import tuple_row
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from ..config import Settings, get_settings
from ..features.regions import REGIONS, regions_of
from ..ingest.store import Store

SPAN = 16
WARMUP = 400
SAMPLES = 800
CHAINS = 2
SEED = 0
HOLDOUT = 0.25
MIN_TRANSITIONS = 60
WORKERS = 8
SUBSAMPLE = 20000
FOLDS = 5
COMPONENTS = 12
MOVEMENT_COLUMNS = [f"move_{index}" for index in range(COMPONENTS)]
KAPPA_BOUNDS = (0.5, 4000.0)

QUERY = (
    "SELECT f.puuid, f.minute, f.x, f.y, p.team_id"
    " FROM frames f JOIN participations p"
    " ON p.match_id = f.match_id AND p.puuid = f.puuid"
    " WHERE f.minute < %s AND f.x IS NOT NULL AND f.match_id = ANY(%s)"
    " ORDER BY f.match_id, f.puuid, f.minute"
)


def _shard(settings: Settings, match_ids: list[str], shard: int) -> tuple:
    store = Store(settings)
    try:
        with store.conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(QUERY, (SPAN, match_ids))
            rows = cursor.fetchall()
    finally:
        store.close()
    if not rows:
        return [], np.zeros((0, len(REGIONS), len(REGIONS)), np.int32), np.zeros(
            (0, len(REGIONS), len(REGIONS)), np.int32
        )
    puuid = np.array([row[0] for row in rows])
    minute = np.fromiter((row[1] for row in rows), np.int16, len(rows))
    x = np.fromiter((row[2] for row in rows), np.float64, len(rows))
    y = np.fromiter((row[3] for row in rows), np.float64, len(rows))
    team = np.fromiter((row[4] or 100 for row in rows), np.int16, len(rows))

    region = regions_of(x, y, team)
    codes, players = pd.factorize(puuid)
    seats = codes.astype(np.int64)
    step = (minute[1:] == minute[:-1] + 1) & (seats[1:] == seats[:-1])
    game = np.concatenate([[0], np.cumsum(~step)])
    source, target = region[:-1][step], region[1:][step]
    who, spell = seats[:-1][step], game[:-1][step]

    size = len(REGIONS)
    train = np.zeros((len(players), size, size), np.int32)
    test = np.zeros((len(players), size, size), np.int32)
    rng = np.random.default_rng(SEED + shard)
    held = rng.random(int(game.max()) + 1)[spell] < HOLDOUT
    np.add.at(train, (who[~held], source[~held], target[~held]), 1)
    np.add.at(test, (who[held], source[held], target[held]), 1)
    return list(players), train, test


def _split(items: list, parts: int) -> list[list]:
    size = (len(items) + parts - 1) // parts
    return [items[index : index + size] for index in range(0, len(items), size)]


def all_matches(settings: Settings) -> list[str]:
    store = Store(settings)
    try:
        with store.conn.cursor() as cursor:
            cursor.execute("SELECT match_id FROM matches ORDER BY match_id")
            return [row["match_id"] for row in cursor.fetchall()]
    finally:
        store.close()


def player_counts(
    settings: Settings, workers: int = WORKERS, match_ids: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    match_ids = list(match_ids) if match_ids is not None else all_matches(settings)
    chunks = _split(match_ids, workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(_shard, [settings] * len(chunks), chunks, range(len(chunks)))
        )

    seats: dict[str, int] = {}
    for players, _, _ in results:
        for puuid in players:
            seats.setdefault(puuid, len(seats))
    size = len(REGIONS)
    train = np.zeros((len(seats), size, size), np.int32)
    test = np.zeros((len(seats), size, size), np.int32)
    for players, part_train, part_test in results:
        if not players:
            continue
        index = np.array([seats[puuid] for puuid in players])
        np.add.at(train, index, part_train)
        np.add.at(test, index, part_test)
    return train, test, sorted(seats, key=seats.get)


def marginal(kappa: float, counts: np.ndarray, totals: np.ndarray, world: np.ndarray) -> float:
    alpha = kappa * world
    total = alpha.sum(axis=1)
    return float(
        (
            gammaln(total)
            - gammaln(totals + total)
            + (gammaln(counts + alpha) - gammaln(alpha)).sum(axis=1)
        ).sum()
    )


def best_kappa(counts: np.ndarray, totals: np.ndarray, world: np.ndarray) -> float:
    found = minimize_scalar(
        lambda log: -marginal(float(np.exp(log)), counts, totals, world),
        bounds=(float(np.log(KAPPA_BOUNDS[0])), float(np.log(KAPPA_BOUNDS[1]))),
        method="bounded",
    )
    return float(np.exp(found.x))


def model(counts, totals, world):
    kappa = numpyro.sample("kappa", dist.LogNormal(3.0, 1.5))
    with numpyro.plate("rows", counts.shape[0]):
        numpyro.sample(
            "counts",
            dist.DirichletMultinomial(kappa * world, total_count=totals),
            obs=counts,
        )


def loglik(probs: np.ndarray, counts: np.ndarray) -> float:
    total = counts.sum()
    if not total:
        return float("nan")
    return float((counts * np.log(np.where(probs > 0, probs, 1e-300))).sum() / total)


def signatures(train: np.ndarray, world: np.ndarray, kappa: float) -> np.ndarray:
    prior = kappa * world
    shrunk = (train + prior[None]) / (
        train.sum(axis=2, keepdims=True) + prior.sum(axis=1)[None, :, None]
    )
    return np.log(shrunk) - np.log(world)[None]


def fit_movement(
    settings: Settings | None = None,
    samples: int = SAMPLES,
    workers: int = WORKERS,
    match_ids: list[str] | None = None,
    report_path: str = "movement_report.json",
    transitions_path: str = "movement_transitions.npz",
) -> dict:
    settings = settings or get_settings()
    train, test, players = player_counts(settings, workers, match_ids)
    size = len(REGIONS)

    every = train + test
    pooled = train.sum(axis=0)
    world = (pooled + 1.0) / (pooled.sum(axis=1, keepdims=True) + size)
    whole = every.sum(axis=0)
    shipped = (whole + 1.0) / (whole.sum(axis=1, keepdims=True) + size)

    rich = train.sum(axis=(1, 2)) >= MIN_TRANSITIONS
    block = train[rich].reshape(-1, size)
    totals = block.sum(axis=1)
    source = np.tile(np.arange(size), int(rich.sum()))
    alive = totals > 0

    counts, weights, rows = block[alive], totals[alive], world[source[alive]]
    kappa = best_kappa(counts, weights, rows)

    picked = np.random.default_rng(SEED).choice(
        len(counts), size=min(SUBSAMPLE, len(counts)), replace=False
    )
    mcmc = MCMC(
        NUTS(model), num_warmup=WARMUP, num_samples=samples,
        num_chains=CHAINS, progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(SEED),
        jnp.asarray(counts[picked]),
        jnp.asarray(weights[picked]),
        jnp.asarray(rows[picked]),
    )
    draws = np.asarray(mcmc.get_samples()["kappa"])
    extra = mcmc_summary(mcmc.get_samples(group_by_chain=True))

    prior = kappa * world
    shrunk = (train + prior[None]) / (
        train.sum(axis=2, keepdims=True) + prior.sum(axis=1)[None, :, None]
    )
    seen = train.sum(axis=2, keepdims=True)
    private = np.divide(train, seen, out=np.zeros(train.shape, float), where=seen > 0)

    check = rich & (test.sum(axis=(1, 2)) > 0)
    held = test[check]
    scores = {
        "world": loglik(np.broadcast_to(world, held.shape), held),
        "player_mle": loglik(private[check], held),
        "player_shrunk": loglik(shrunk[check], held),
    }
    report = {
        "players": int(len(players)),
        "players_scored": int(check.sum()),
        "min_transitions": MIN_TRANSITIONS,
        "train_transitions": int(train.sum()),
        "holdout_transitions": int(test.sum()),
        "rows_fitted": int(alive.sum()),
        "workers": workers,
        "kappa": round(kappa, 3),
        "kappa_subsample": {
            "rows": int(len(picked)),
            "median": round(float(np.median(draws)), 3),
            "low": round(float(np.quantile(draws, 0.025)), 3),
            "high": round(float(np.quantile(draws, 0.975)), 3),
        },
        "r_hat": round(float(np.nanmax(extra["kappa"]["r_hat"])), 4),
        "n_eff": round(float(np.nanmin(extra["kappa"]["n_eff"])), 1),
        "loglik": {name: round(value, 5) for name, value in scores.items()},
        "lift_over_world": round(scores["player_shrunk"] - scores["world"], 5),
        "world_self_transition": round(float(np.diag(world).mean()), 4),
        "world_entropy_bits": round(
            float(-(world * np.log2(np.where(world > 0, world, 1.0))).sum(axis=1).mean()), 4
        ),
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_dir / report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez(
        settings.model_dir / transitions_path,
        world=shipped, kappa=np.array([kappa]), players=np.array(players),
        signature=signatures(every, shipped, kappa)
        .reshape(len(players), -1)
        .astype(np.float32),
        seen=every.sum(axis=(1, 2)),
    )
    return report


def movement_features(
    settings: Settings | None = None, folds: int = FOLDS, workers: int = WORKERS
) -> pd.DataFrame:
    settings = settings or get_settings()
    matches = all_matches(settings)
    order = np.random.default_rng(SEED).permutation(len(matches))
    parts = np.array_split(order, folds)

    report = fit_movement(
        settings, workers=workers,
        report_path="movement_report.json", transitions_path="movement_transitions.npz",
    )
    store = np.load(settings.model_dir / "movement_transitions.npz", allow_pickle=True)
    base = store["signature"][store["seen"] >= MIN_TRANSITIONS]
    centre = base.mean(axis=0)
    _, _, basis = np.linalg.svd(base - centre, full_matrices=False)
    basis = basis[:COMPONENTS]

    rows = []
    for index, part in enumerate(parts):
        inside = {matches[i] for i in part}
        outside = [m for m in matches if m not in inside]
        fit_movement(
            settings, workers=workers, match_ids=outside,
            report_path=f"movement_fold_{index}.json",
            transitions_path=f"movement_fold_{index}.npz",
        )
        fold = np.load(settings.model_dir / f"movement_fold_{index}.npz", allow_pickle=True)
        rich = fold["seen"] >= MIN_TRANSITIONS
        scores = (fold["signature"][rich] - centre) @ basis.T
        frame = pd.DataFrame(scores, columns=MOVEMENT_COLUMNS)
        frame["puuid"] = fold["players"][rich]
        frame["fold"] = index
        rows.append(frame)

    table = pd.concat(rows, ignore_index=True)
    assignment = pd.DataFrame(
        {
            "match_id": [matches[i] for part in parts for i in part],
            "fold": [index for index, part in enumerate(parts) for _ in part],
        }
    )
    seats = pd.read_parquet(
        settings.processed_dir / "participations.parquet", columns=["match_id", "puuid"]
    )
    joined = seats.merge(assignment, on="match_id", how="inner").merge(
        table, on=["fold", "puuid"], how="inner"
    )
    joined.drop(columns=["fold"]).to_parquet(
        settings.processed_dir / "movement.parquet", index=False
    )
    np.savez(
        settings.model_dir / "movement_basis.npz", basis=basis, centre=centre
    )
    return report


def load_movement(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_dir / "movement.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "puuid", *MOVEMENT_COLUMNS])
    return pd.read_parquet(path)
