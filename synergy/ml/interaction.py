import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
from numpyro.infer import SVI, Trace_ELBO, autoguide

from ..config import Settings, get_settings
from ..deep.stream import SEATS
from ..deep.walk import MatchWalk, batches
from .gold import team_advantage

SOURCES = ("style", "walk")

COMPONENTS = 16
STEPS = 8000
MAX_STEPS = 120000
LEARNING_RATE = 0.005
TOLERANCE = 1e-4
WINDOW = 400
SEED = 0
HOLDOUT = 0.2
NULLS = 40
BATCH = 128
TEAM_SIZE = 5
TEAM_PAIRS = TEAM_SIZE * (TEAM_SIZE - 1) // 2
TERMS = ("linear", "solo", "pair")


def seat_matrix(settings: Settings, stream: str = "stream.npz", cache: bool = True) -> dict:
    kept = settings.processed_dir / f"seats_{stream}"
    if cache and kept.exists():
        stored = np.load(kept, allow_pickle=True)
        print(f"seat encodings from cache {kept.name}", flush=True)
        return {key: stored[key] for key in stored.files}
    started = time.time()
    raw = np.load(settings.processed_dir / stream, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MatchWalk(
        kinds=int(len(data["kinds"])),
        regions=int(len(data["regions"])),
        champions=int(len(data["champions"])),
        span=int(data["kind"].shape[1]),
    ).to(device)
    model.load_state_dict(torch.load(settings.model_dir / "match_walk.pt", map_location=device))
    model.eval()
    blocks = []
    with torch.no_grad():
        for *inputs, _ in batches(
            data, np.arange(len(data["match_id"])), BATCH, device, shuffle=False
        ):
            blocks.append(model.seat_encodings(*inputs).float().cpu().numpy())
    built = {
        "match_id": data["match_id"],
        "seat_puuid": data["seat_puuid"],
        "seat_side": data["seat_side"],
        "encoding": np.concatenate(blocks),
    }
    print(f"encoded {len(built['match_id']):,} matches in {time.time() - started:.0f}s", flush=True)
    if cache:
        np.savez(kept, **built)
    return built


def style_matrix(settings: Settings, stream: str = "stream.npz") -> dict:
    from ..features.habit import HABIT_COLUMNS, load_habits
    from ..features.orphans import ORPHAN_COLUMNS, load_orphan_features
    from ..features.tendency import TENDENCY_COLUMNS, load_tendencies
    from .embedding import EMBED_COLUMNS, load_embedding
    from .movement import MOVEMENT_COLUMNS, load_movement

    raw = np.load(settings.processed_dir / stream, allow_pickle=True)
    seats = pd.DataFrame(
        {"match_id": np.repeat(raw["match_id"], SEATS), "puuid": raw["seat_puuid"].ravel()}
    )
    blocks = (
        (load_tendencies(settings), TENDENCY_COLUMNS),
        (load_orphan_features(settings), ORPHAN_COLUMNS),
        (load_habits(settings), HABIT_COLUMNS),
        (load_movement(settings), MOVEMENT_COLUMNS),
        (load_embedding(settings), EMBED_COLUMNS),
    )
    columns = []
    for table, names in blocks:
        if table.empty:
            continue
        seats = seats.merge(table[["match_id", "puuid", *names]], on=["match_id", "puuid"], how="left")
        columns.extend(names)
    values = seats[columns].to_numpy(dtype=float)
    values = (values - np.nanmean(values, axis=0)) / np.nanstd(values, axis=0).clip(min=1e-6)
    values = np.nan_to_num(values, nan=0.0)
    print(f"style blocks: {len(columns)} leave one out columns, "
          f"{float(seats[columns].notna().all(axis=1).mean()):.3f} of seats fully covered", flush=True)
    return {
        "match_id": raw["match_id"],
        "seat_puuid": raw["seat_puuid"],
        "seat_side": raw["seat_side"],
        "encoding": values.reshape(len(raw["match_id"]), SEATS, len(columns)).astype(np.float32),
        "columns": np.array(columns),
    }


def reduce(encoding: np.ndarray, fit: np.ndarray, components: int = COMPONENTS) -> np.ndarray:
    seen = encoding[fit].reshape(-1, encoding.shape[-1])
    centre = seen.mean(axis=0)
    _, _, basis = np.linalg.svd(seen - centre, full_matrices=False)
    taken = basis[:components]
    spread = ((seen - centre) @ taken.T).std(axis=0, keepdims=True).clip(min=1e-6)
    flat = (encoding.reshape(-1, encoding.shape[-1]) - centre) @ taken.T / spread
    return flat.reshape(encoding.shape[0], encoding.shape[1], components).astype(np.float32)


def sides(seat_side: np.ndarray, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mine = seat_side == 1
    if not (mine.sum(axis=1) == TEAM_SIZE).all():
        raise ValueError("every match must have exactly five blue seats")
    rows = np.arange(len(reduced))[:, None]
    blue = reduced[rows, np.argsort(~mine, axis=1, kind="stable")[:, :TEAM_SIZE]]
    red = reduced[rows, np.argsort(mine, axis=1, kind="stable")[:, :TEAM_SIZE]]
    return blue, red


def _selves(team, matrix, components):
    return jnp.einsum("bsi,ij,bsj->b", team, matrix, team) / (TEAM_SIZE * components)


def _crossed(team, matrix, components):
    total = team.sum(axis=1)
    whole = jnp.einsum("bi,ij,bj->b", total, matrix, total)
    selves = jnp.einsum("bsi,ij,bsj->b", team, matrix, team)
    return 0.5 * (whole - selves) / (TEAM_PAIRS * components)


def _square(name, components, scale):
    with numpyro.plate(f"{name}_rows", components):
        with numpyro.plate(f"{name}_columns", components):
            raw = numpyro.sample(name, dist.Normal(0.0, scale))
    return 0.5 * (raw + raw.T)


def model(blue, red, components, terms: str = "pair", gold=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    main_scale = numpyro.sample("main_scale", dist.HalfNormal(1.0))
    residual = numpyro.sample("residual", dist.HalfNormal(1.0))
    with numpyro.plate("axes", components):
        weight = numpyro.sample("weight", dist.Normal(0.0, main_scale))
    centre = mean + (blue.sum(axis=1) - red.sum(axis=1)) @ weight
    if terms in ("solo", "pair"):
        solo = _square("solo", components, numpyro.sample("solo_scale", dist.HalfNormal(1.0)))
        centre = centre + _selves(blue, solo, components) - _selves(red, solo, components)
    if terms == "pair":
        cross = _square("cross", components, numpyro.sample("cross_scale", dist.HalfNormal(1.0)))
        centre = centre + _crossed(blue, cross, components) - _crossed(red, cross, components)
    with numpyro.plate("matches", centre.shape[0]):
        numpyro.sample("gold", dist.Normal(centre, residual), obs=gold)


def _settled(losses: np.ndarray) -> float:
    if len(losses) < 2 * WINDOW:
        return float("inf")
    late = losses[-WINDOW:].mean()
    early = losses[-2 * WINDOW : -WINDOW].mean()
    return float(abs(early - late) / max(abs(late), 1.0))


def _predict(drawn, blue, red, components, terms):
    centre = drawn["mean"] + (blue.sum(axis=1) - red.sum(axis=1)) @ drawn["weight"]
    cross = None
    if terms in ("solo", "pair"):
        solo = 0.5 * (drawn["solo"] + drawn["solo"].T)
        centre = centre + _selves(blue, solo, components) - _selves(red, solo, components)
    if terms == "pair":
        cross = 0.5 * (drawn["cross"] + drawn["cross"].T)
        centre = centre + _crossed(blue, cross, components) - _crossed(red, cross, components)
    return centre, cross


def _fit(blue, red, target, fit, test, components, terms, steps, seed, max_steps=MAX_STEPS):
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(LEARNING_RATE), Trace_ELBO())
    arguments = (jnp.asarray(blue[fit]), jnp.asarray(red[fit]), components)
    options = {"terms": terms, "gold": jnp.asarray(target[fit]), "progress_bar": False}
    result = svi.run(jax.random.PRNGKey(seed), steps, *arguments, **options)
    losses, taken = np.asarray(result.losses), steps
    while _settled(losses) > TOLERANCE and taken < max_steps:
        result = svi.run(
            jax.random.PRNGKey(seed), steps, *arguments, init_state=result.state, **options
        )
        losses = np.concatenate([losses, np.asarray(result.losses)])
        taken += steps

    drawn = guide.median(result.params)
    centre, cross = _predict(
        drawn, jnp.asarray(blue[test]), jnp.asarray(red[test]), components, terms
    )
    held = target[test]
    error = float(((np.asarray(centre) - held) ** 2).sum())
    return {
        "r2": 1.0 - error / float(((held - held.mean()) ** 2).sum()),
        "steps": taken,
        "loss": float(losses[-WINDOW:].mean()),
        "drift": round(_settled(losses), 7),
        "settled": bool(_settled(losses) <= TOLERANCE),
        "matrix": np.asarray(cross) if cross is not None else None,
    }


def _rungs(blue, red, target, fit, test, components, steps, seed, names=TERMS):
    out = {}
    for name in names:
        clock = time.time()
        out[name] = _fit(blue, red, target, fit, test, components, name, steps, seed)
        print(
            f"  {name:7} r2 {out[name]['r2']:.6f}  loss {out[name]['loss']:.0f}"
            f"  steps {out[name]['steps']}  drift {out[name]['drift']:.2e}"
            f"  {'settled' if out[name]['settled'] else 'NOT SETTLED'}"
            f"  {time.time() - clock:.0f}s",
            flush=True,
        )
    return out


def _basis(settings: Settings, stream: str, components: int, seed: int, source: str) -> dict:
    seats = style_matrix(settings, stream) if source == "style" else seat_matrix(settings, stream)
    gold = team_advantage(settings).reindex(seats["match_id"])
    keep = gold.notna().to_numpy()
    raw = gold.to_numpy(dtype=float)[keep]
    order = np.random.default_rng(seed).permutation(len(raw))
    cut = int(len(raw) * (1.0 - HOLDOUT))
    return {
        "match_id": seats["match_id"][keep],
        "seat_puuid": seats["seat_puuid"][keep],
        "seat_side": seats["seat_side"][keep],
        "reduced": reduce(seats["encoding"][keep], order[:cut], components),
        "raw": raw,
        "fit": order[:cut],
        "test": order[cut:],
    }


def fit_interaction(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    components: int = COMPONENTS,
    steps: int = STEPS,
    nulls: int = NULLS,
    seed: int = SEED,
    source: str = "style",
    report_path: str = "interaction_report.json",
) -> dict:
    settings = settings or get_settings()
    basis = _basis(settings, stream, components, seed, source)
    raw, fit, test = basis["raw"], basis["fit"], basis["test"]
    target = (raw - raw[fit].mean()) / raw[fit].std()
    blue, red = sides(basis["seat_side"], basis["reduced"])

    print(f"fitting {len(fit):,} matches, holding out {len(test):,}", flush=True)
    rungs = _rungs(blue, red, target, fit, test, components, steps, seed)
    if rungs["pair"]["loss"] > rungs["solo"]["loss"]:
        raise ValueError(
            f"pair nests solo but fitted worse, {rungs['pair']['loss']:.0f}"
            f" against {rungs['solo']['loss']:.0f}, so the optimiser failed"
        )
    gain = rungs["pair"]["r2"] - rungs["solo"]["r2"]
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(settings.model_dir / "interaction_matrix.npz", matrix=rungs["pair"]["matrix"])
    print(f"  gain {gain:.6f}, matrix saved, now {nulls} nulls", flush=True)

    draws = []
    for draw in range(nulls):
        clock = time.time()
        rng = np.random.default_rng(1000 + draw)
        mixed_blue = np.stack([blue[rng.permutation(len(blue)), seat] for seat in range(TEAM_SIZE)], axis=1)
        mixed_red = np.stack([red[rng.permutation(len(red)), seat] for seat in range(TEAM_SIZE)], axis=1)
        under = _rungs(
            mixed_blue, mixed_red, target, fit, test, components, steps, seed, names=("solo", "pair")
        )
        draws.append(under["pair"]["r2"] - under["solo"]["r2"])
        print(
            f"  null {draw + 1}/{nulls} gain {draws[-1]:.6f}"
            f"  above {int(np.sum(np.array(draws) >= gain))}  {time.time() - clock:.0f}s",
            flush=True,
        )

    report = {
        "source": source,
        "z": "leave one out playstyle blocks from the player's other games" if source == "style"
        else "per seat attention pooling from the walk over this match's events",
        "matches": int(len(target)),
        "held_out": int(len(test)),
        "components": components,
        "rungs": {
            name: {key: value for key, value in rung.items() if key != "matrix"}
            for name, rung in rungs.items()
        },
        "solo_over_linear": round(rungs["solo"]["r2"] - rungs["linear"]["r2"], 6),
        "gain": round(gain, 6),
        "settled": all(rung["settled"] for rung in rungs.values()),
        "null": "seats reshuffled across matches within role and side, target kept",
        "nulls": nulls,
    }
    if draws:
        drawn = np.array(draws)
        report |= {
            "null_mean": round(float(drawn.mean()), 6),
            "null_sd": round(float(drawn.std()), 6),
            "nulls_above": int((drawn >= gain).sum()),
        }
    with open(settings.model_dir / report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def pair_scores(
    settings: Settings | None = None, stream: str = "stream.npz", source: str = "style"
) -> pd.DataFrame:
    settings = settings or get_settings()
    matrix = np.load(settings.model_dir / "interaction_matrix.npz")["matrix"]
    basis = _basis(settings, stream, matrix.shape[0], SEED, source)
    reduced, components = basis["reduced"], matrix.shape[0]
    left, right = np.triu_indices(TEAM_SIZE, k=1)
    frames = []
    for picks in (basis["seat_side"] == 1, basis["seat_side"] == 0):
        seated = np.argsort(~picks, axis=1, kind="stable")[:, :TEAM_SIZE]
        rows = np.arange(len(reduced))[:, None]
        team = reduced[rows, seated]
        scored = np.einsum("bpi,ij,bqj->bpq", team, matrix, team) / (TEAM_PAIRS * components)
        frames.append(
            pd.DataFrame(
                {
                    "match_id": np.repeat(basis["match_id"], len(left)),
                    "puuid_a": basis["seat_puuid"][rows, seated][:, left].ravel(),
                    "puuid_b": basis["seat_puuid"][rows, seated][:, right].ravel(),
                    "synergy": scored[:, left, right].ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def write_scores(
    settings: Settings | None = None, stream: str = "stream.npz", source: str = "style"
) -> dict:
    settings = settings or get_settings()
    matrix = np.load(settings.model_dir / "interaction_matrix.npz")["matrix"]
    basis = _basis(settings, stream, matrix.shape[0], SEED, source)
    flat = pd.DataFrame(
        basis["reduced"].reshape(-1, matrix.shape[0]),
        columns=[f"c{index}" for index in range(matrix.shape[0])],
    )
    flat["puuid"] = basis["seat_puuid"].ravel()
    grouped = flat.groupby("puuid")
    styles = grouped.mean()
    styles["seats"] = grouped.size()
    styles.reset_index().to_parquet(settings.processed_dir / "player_styles.parquet", index=False)
    pairs = pair_scores(settings, stream, source)
    pairs.to_parquet(settings.processed_dir / "pair_synergy.parquet", index=False)
    quantiles = np.quantile(pairs["synergy"].to_numpy(), np.linspace(0.0, 1.0, 1001))
    np.savez(settings.model_dir / "interaction_scores.npz", quantiles=quantiles)
    return {
        "players": int(len(styles)),
        "pair_rows": int(len(pairs)),
        "synergy_sd": round(float(pairs["synergy"].std()), 6),
    }


def pair_between(left: str, right: str, settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    matrix_path = settings.model_dir / "interaction_matrix.npz"
    styles_path = settings.processed_dir / "player_styles.parquet"
    scores_path = settings.model_dir / "interaction_scores.npz"
    if not (matrix_path.exists() and styles_path.exists() and scores_path.exists()):
        return None
    matrix = np.load(matrix_path)["matrix"]
    styles = pd.read_parquet(styles_path).set_index("puuid")
    if left not in styles.index or right not in styles.index:
        return None
    columns = [f"c{index}" for index in range(matrix.shape[0])]
    a, b = styles.loc[left, columns].to_numpy(dtype=float), styles.loc[right, columns].to_numpy(dtype=float)
    value = float(a @ matrix @ b) / (TEAM_PAIRS * matrix.shape[0])
    quantiles = np.load(scores_path)["quantiles"]
    return {
        "synergy": round(value, 6),
        "percentile": round(float(np.searchsorted(quantiles, value) / 10.0), 1),
        "left_seats": int(styles.loc[left, "seats"]),
        "right_seats": int(styles.loc[right, "seats"]),
    }
