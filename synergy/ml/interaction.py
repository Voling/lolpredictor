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
from ..features.positions import KEY, UNKNOWN
from .gold import team_advantage
from .serving import TEAM_PAIRS, TEAM_SIZE

SOURCES = ("style", "walk")
RANK = 8
STEPS = 8000
MAX_STEPS = 120000
LEARNING_RATE = 0.005
TOLERANCE = 1e-4
NESTING_TOLERANCE = 1e-4
WINDOW = 4000
SEED = 0
HOLDOUT = 0.2
NULLS = 40
BATCH = 128
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
        "columns": np.array([f"walk_{index}" for index in range(blocks[0].shape[-1])]),
    }
    print(f"encoded {len(built['match_id']):,} matches in {time.time() - started:.0f}s", flush=True)
    if cache:
        np.savez(kept, **built)
    return built


def style_matrix(settings: Settings, stream: str = "stream.npz") -> dict:
    from ..features.habit import HABIT_COLUMNS, load_habits
    from ..features.priority import PRIORITY_COLUMNS, load_priority
    from ..features.reaction import REACTION_COLUMNS, load_reaction
    from ..features.tendency import TENDENCY_COLUMNS, load_tendencies
    from .embedding import EMBED_COLUMNS, load_embedding
    from .movement import MOVEMENT_COLUMNS, load_movement

    raw = np.load(settings.processed_dir / stream, allow_pickle=True)
    seats = pd.DataFrame(
        {"match_id": np.repeat(raw["match_id"], SEATS), "puuid": raw["seat_puuid"].ravel()}
    )
    loaders = (
        (load_tendencies, TENDENCY_COLUMNS),
        (load_priority, PRIORITY_COLUMNS),
        (load_reaction, REACTION_COLUMNS),
        (load_habits, HABIT_COLUMNS),
        (load_movement, MOVEMENT_COLUMNS),
        (load_embedding, EMBED_COLUMNS),
    )
    values = np.full((len(seats), sum(len(names) for _, names in loaders)), np.nan, dtype=np.float32)
    covered = np.ones(len(seats), bool)
    columns, filled, start = [], [], 0
    for load, names in loaders:
        table = load(settings)
        if not table.empty:
            slot = seats.merge(
                table[["match_id", "puuid"]].assign(slot=np.arange(len(table))), on=["match_id", "puuid"], how="left"
            )["slot"].to_numpy(dtype=float)
            if len(slot) != len(seats):
                raise ValueError(f"a style block repeats a seat, {len(slot):,} rows for {len(seats):,} seats")
            seen = ~np.isnan(slot)
            rows = slot[seen].astype(np.int64)
            for offset, name in enumerate(names):
                column = table[name].to_numpy(dtype=float)[rows]
                values[seen, start + offset] = column
                covered[seen] &= ~np.isnan(column)
            if names:
                covered &= seen
            columns.extend(names)
            filled.extend(range(start, start + len(names)))
        start += len(names)
        del table
    if len(filled) < values.shape[1]:
        values = values[:, filled]
    print(f"style blocks: {len(columns)} leave one out columns, "
          f"{float(covered.mean()):.3f} of seats fully covered", flush=True)
    return {
        "match_id": raw["match_id"],
        "seat_puuid": raw["seat_puuid"],
        "seat_side": raw["seat_side"],
        "encoding": values.reshape(len(raw["match_id"]), SEATS, len(columns)),
        "columns": np.array(columns),
    }


def seat_positions(settings: Settings, match_id: np.ndarray, seat_puuid: np.ndarray) -> np.ndarray:
    seats = pd.DataFrame({"match_id": np.repeat(match_id, SEATS), "puuid": seat_puuid.ravel()})
    seated = pd.read_parquet(settings.processed_dir / "participations.parquet", columns=["match_id", *KEY])
    seats = seats.merge(seated, on=["match_id", "puuid"], how="left")
    return seats["position"].fillna(UNKNOWN).to_numpy().reshape(len(match_id), SEATS)


def fully_seated(positions: np.ndarray) -> np.ndarray:
    return (positions != UNKNOWN).all(axis=1)


def _moments(encoding: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen = encoding[fit].reshape(-1, encoding.shape[-1])
    return np.nanmean(seen, axis=0), np.nanstd(seen, axis=0).clip(min=1e-6)


def standardise(encoding: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre, spread = _moments(encoding, fit)
    flat = encoding.reshape(-1, encoding.shape[-1]) - centre
    flat /= spread
    np.nan_to_num(flat, copy=False, nan=0.0)
    return flat.reshape(encoding.shape).astype(np.float32, copy=False), centre, spread


def sides(seat_side: np.ndarray, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mine = seat_side == 1
    if not (mine.sum(axis=1) == TEAM_SIZE).all():
        raise ValueError("every match must have exactly five blue seats")
    rows = np.arange(len(reduced))[:, None]
    blue = reduced[rows, np.argsort(~mine, axis=1, kind="stable")[:, :TEAM_SIZE]]
    red = reduced[rows, np.argsort(mine, axis=1, kind="stable")[:, :TEAM_SIZE]]
    return blue, red


def shuffle_seats(team: np.ndarray, rng: np.random.Generator, out: np.ndarray) -> np.ndarray:
    for seat in range(TEAM_SIZE):
        out[:, seat] = team[rng.permutation(len(team)), seat]
    return out


def _selves(team, factor, dim):
    loading, weight = factor
    projected = jnp.einsum("bsi,ri->bsr", team, loading)
    return (projected**2 * weight).sum(axis=(1, 2)) / (TEAM_SIZE * dim)


def _crossed(team, factor, dim):
    loading, weight = factor
    projected = jnp.einsum("bsi,ri->bsr", team, loading)
    whole = (projected.sum(axis=1) ** 2 * weight).sum(axis=1)
    selves = (projected**2 * weight).sum(axis=(1, 2))
    return 0.5 * (whole - selves) / (TEAM_PAIRS * dim)


def _factored(name, dim, rank, scale):
    with numpyro.plate(f"{name}_rows", dim):
        with numpyro.plate(f"{name}_rank", rank):
            loading = numpyro.sample(f"{name}_loading", dist.Normal(0.0, 1.0))
    with numpyro.plate(f"{name}_axes", rank):
        weight = numpyro.sample(f"{name}_weight", dist.Normal(0.0, scale))
    return loading, weight


def model(blue, red, dim, rank: int = RANK, terms: str = "pair", gold=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    main_scale = numpyro.sample("main_scale", dist.HalfNormal(1.0))
    residual = numpyro.sample("residual", dist.HalfNormal(1.0))
    with numpyro.plate("axes", dim):
        weight = numpyro.sample("weight", dist.Normal(0.0, main_scale))
    centre = mean + (blue.sum(axis=1) - red.sum(axis=1)) @ weight / dim
    if terms in ("solo", "pair"):
        solo = _factored("solo", dim, rank, numpyro.sample("solo_scale", dist.HalfNormal(1.0)))
        centre = centre + _selves(blue, solo, dim) - _selves(red, solo, dim)
    if terms == "pair":
        cross = _factored("cross", dim, rank, numpyro.sample("cross_scale", dist.HalfNormal(1.0)))
        centre = centre + _crossed(blue, cross, dim) - _crossed(red, cross, dim)
    with numpyro.plate("matches", centre.shape[0]):
        numpyro.sample("gold", dist.Normal(centre, residual), obs=gold)


def _dense(drawn: dict, name: str) -> jnp.ndarray:
    loading, weight = drawn[f"{name}_loading"], drawn[f"{name}_weight"]
    return (loading * weight[:, None]).T @ loading


def _settled(losses: np.ndarray) -> float:
    if len(losses) < 2 * WINDOW:
        return float("inf")
    late = losses[-WINDOW:].mean()
    early = losses[-2 * WINDOW : -WINDOW].mean()
    return float(abs(early - late) / max(abs(late), 1.0))


def _predict(drawn, blue, red, dim, terms):
    centre = drawn["mean"] + (blue.sum(axis=1) - red.sum(axis=1)) @ drawn["weight"] / dim
    cross = None
    if terms in ("solo", "pair"):
        solo = (drawn["solo_loading"], drawn["solo_weight"])
        centre = centre + _selves(blue, solo, dim) - _selves(red, solo, dim)
    if terms == "pair":
        factor = (drawn["cross_loading"], drawn["cross_weight"])
        centre = centre + _crossed(blue, factor, dim) - _crossed(red, factor, dim)
        cross = _dense(drawn, "cross")
    return centre, cross


def _fit(blue, red, target, fit, test, rank, terms, steps, seed, max_steps=MAX_STEPS, min_steps=0):
    dim = blue.shape[-1]
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(LEARNING_RATE), Trace_ELBO())
    arguments = (jnp.asarray(blue[fit]), jnp.asarray(red[fit]), dim)
    options = {"rank": rank, "terms": terms, "gold": jnp.asarray(target[fit]), "progress_bar": False}
    seen = target[fit]
    baseline = float(((seen - seen.mean()) ** 2).sum())

    def fit_r2(params):
        fitted, _ = _predict(guide.median(params), *arguments, terms)
        return 1.0 - float(((np.asarray(fitted) - seen) ** 2).sum()) / baseline

    result = svi.run(jax.random.PRNGKey(seed), steps, *arguments, **options)
    losses, taken = np.asarray(result.losses), steps
    blocks = [fit_r2(result.params)]
    while (_settled(losses) > TOLERANCE or taken < min_steps) and taken < max_steps:
        result = svi.run(
            jax.random.PRNGKey(seed), steps, *arguments, init_state=result.state, **options
        )
        losses = np.concatenate([losses, np.asarray(result.losses)])
        taken += steps
        blocks.append(fit_r2(result.params))

    drawn = guide.median(result.params)
    centre, cross = _predict(drawn, jnp.asarray(blue[test]), jnp.asarray(red[test]), dim, terms)
    held = target[test]
    error = float(((np.asarray(centre) - held) ** 2).sum())
    return {
        "r2": 1.0 - error / float(((held - held.mean()) ** 2).sum()),
        "fit_r2": blocks[-1],
        "fit_blocks": [round(value, 6) for value in blocks],
        "steps": taken,
        "loss": float(losses[-WINDOW:].mean()),
        "drift": round(_settled(losses), 7),
        "settled": bool(_settled(losses) <= TOLERANCE),
        "matrix": np.asarray(cross) if cross is not None else None,
    }


def _rungs(blue, red, target, fit, test, rank, steps, seed, names=TERMS, floor=0):
    out = {}
    for name in names:
        clock = time.time()
        out[name] = _fit(blue, red, target, fit, test, rank, name, steps, seed, min_steps=floor)
        floor = out[name]["steps"]
        print(
            f"  {name:7} r2 {out[name]['r2']:.6f}  fit r2 {out[name]['fit_r2']:.6f}  loss {out[name]['loss']:.0f}"
            f"  steps {out[name]['steps']}  drift {out[name]['drift']:.2e}"
            f"  {'settled' if out[name]['settled'] else 'NOT SETTLED'}"
            f"  {time.time() - clock:.0f}s",
            flush=True,
        )
    return out


def _basis(settings: Settings, stream: str, seed: int, source: str) -> dict:
    seats = style_matrix(settings, stream) if source == "style" else seat_matrix(settings, stream)
    positions = seat_positions(settings, seats["match_id"], seats["seat_puuid"])
    gold = team_advantage(settings).reindex(seats["match_id"])
    unseated = ~fully_seated(positions)
    if unseated.any():
        print(f"dropping {int(unseated.sum()):,} matches whose seats have no position", flush=True)
    keep = gold.notna().to_numpy() & ~unseated
    raw = gold.to_numpy(dtype=float)[keep]
    order = np.random.default_rng(seed).permutation(len(raw))
    cut = int(len(raw) * (1.0 - HOLDOUT))
    reduced, centre, spread = standardise(seats.pop("encoding")[keep], order[:cut])
    return {
        "match_id": seats["match_id"][keep],
        "seat_puuid": seats["seat_puuid"][keep],
        "seat_side": seats["seat_side"][keep],
        "seat_position": positions[keep],
        "columns": seats["columns"],
        "reduced": reduced,
        "centre": centre,
        "spread": spread,
        "raw": raw,
        "fit": order[:cut],
        "test": order[cut:],
    }


def fit_interaction(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    rank: int = RANK,
    steps: int = STEPS,
    nulls: int = NULLS,
    seed: int = SEED,
    source: str = "style",
    min_steps: int = 0,
    report_path: str = "interaction_report.json",
) -> dict:
    settings = settings or get_settings()
    basis = _basis(settings, stream, seed, source)
    raw, fit, test = basis["raw"], basis["fit"], basis["test"]
    target = (raw - raw[fit].mean()) / raw[fit].std()
    blue, red = sides(basis["seat_side"], basis.pop("reduced"))
    dim = blue.shape[-1]

    print(f"fitting {len(fit):,} matches, holding out {len(test):,}, {dim} named columns, rank {rank}", flush=True)
    rungs = _rungs(blue, red, target, fit, test, rank, steps, seed, floor=min_steps)
    nested = bool(rungs["pair"]["fit_r2"] >= rungs["solo"]["fit_r2"] - NESTING_TOLERANCE)
    gain = rungs["pair"]["r2"] - rungs["solo"]["r2"]
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        settings.model_dir / "interaction_matrix.npz",
        matrix=rungs["pair"]["matrix"], columns=basis["columns"], centre=basis["centre"], spread=basis["spread"],
    )
    if not nested:
        print(
            f"  pair fits the training matches worse than solo, {rungs['pair']['fit_r2']:.6f} against"
            f" {rungs['solo']['fit_r2']:.6f}: the cross term is shrunk away and adds noise, no nulls run",
            flush=True,
        )
    elif gain <= 0.0:
        print(f"  gain {gain:.6f} is not positive, matrix saved, no nulls run", flush=True)
    else:
        print(f"  gain {gain:.6f}, matrix saved, now {nulls} nulls", flush=True)

    draws = []
    runs = nulls if nested and gain > 0.0 else 0
    mixed_blue, mixed_red = (np.empty_like(blue), np.empty_like(red)) if runs > 0 else (None, None)
    for draw in range(runs):
        clock = time.time()
        rng = np.random.default_rng(1000 + draw)
        shuffle_seats(blue, rng, mixed_blue)
        shuffle_seats(red, rng, mixed_red)
        under = _rungs(
            mixed_blue, mixed_red, target, fit, test, rank, steps, seed, names=("solo", "pair"), floor=min_steps
        )
        draws.append(under["pair"]["r2"] - under["solo"]["r2"])
        print(
            f"  null {draw + 1}/{nulls} gain {draws[-1]:.6f}"
            f"  above {int(np.sum(np.array(draws) >= gain))}  {time.time() - clock:.0f}s",
            flush=True,
        )

    report = {
        "source": source,
        "z": "leave one out playstyle cells from the player's other games" if source == "style"
        else "per seat attention pooling from the walk over this match's events",
        "matches": int(len(target)),
        "held_out": int(len(test)),
        "columns": int(dim),
        "rank": rank,
        "rungs": {
            name: {key: value for key, value in rung.items() if key != "matrix"}
            for name, rung in rungs.items()
        },
        "min_steps": min_steps,
        "solo_over_linear": round(rungs["solo"]["r2"] - rungs["linear"]["r2"], 6),
        "gain": round(gain, 6),
        "nested": nested,
        "settled": all(rung["settled"] for rung in rungs.values()),
        "null": "seats reshuffled across matches within role and side, target kept",
        "nulls": len(draws),
        "informative": False,
    }
    if draws:
        drawn = np.array(draws)
        report |= {
            "null_mean": round(float(drawn.mean()), 6),
            "null_sd": round(float(drawn.std()), 6),
            "nulls_above": int((drawn >= gain).sum()),
            "informative": bool(nested and gain > 0.0 and (drawn >= gain).sum() == 0),
        }
    with open(settings.model_dir / report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _saved(settings: Settings) -> dict:
    stored = np.load(settings.model_dir / "interaction_matrix.npz", allow_pickle=True)
    return {key: stored[key] for key in stored.files}


def pair_scores(
    settings: Settings | None = None, stream: str = "stream.npz", source: str = "style", basis: dict | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    matrix = _saved(settings)["matrix"]
    basis = basis if basis is not None else _basis(settings, stream, SEED, source)
    frames = [_side_pairs(basis, matrix, side) for side in (1, 0)]
    return pd.concat(frames, ignore_index=True)


def _side_pairs(basis: dict, matrix: np.ndarray, side: int) -> pd.DataFrame:
    reduced = basis["reduced"]
    dim = matrix.shape[0]
    left, right = np.triu_indices(TEAM_SIZE, k=1)
    seated = np.argsort(~(basis["seat_side"] == side), axis=1, kind="stable")[:, :TEAM_SIZE]
    rows = np.arange(len(reduced))[:, None]
    team = reduced[rows, seated]
    scored = np.einsum("bpi,ij,bqj->bpq", team, matrix, team) / (TEAM_PAIRS * dim)
    del team
    puuids, positions = basis["seat_puuid"][rows, seated], basis["seat_position"][rows, seated]
    return pd.DataFrame(
        {
            "match_id": np.repeat(basis["match_id"], len(left)),
            "side": side,
            "puuid_a": puuids[:, left].ravel(),
            "position_a": positions[:, left].ravel(),
            "puuid_b": puuids[:, right].ravel(),
            "position_b": positions[:, right].ravel(),
            "synergy": scored[:, left, right].ravel(),
        }
    )


def player_styles(basis: dict, columns: list[str]) -> pd.DataFrame:
    flat = pd.DataFrame(basis["reduced"].reshape(-1, len(columns)), columns=columns)
    flat["puuid"] = basis["seat_puuid"].ravel()
    flat["position"] = basis["seat_position"].ravel()
    grouped = flat.groupby(KEY)
    styles = grouped.mean()
    styles["seats"] = grouped.size()
    return styles


def write_scores(
    settings: Settings | None = None, stream: str = "stream.npz", source: str = "style"
) -> dict:
    settings = settings or get_settings()
    basis = _basis(settings, stream, SEED, source)
    columns = [str(c) for c in basis["columns"]]
    styles = player_styles(basis, columns).join(evidence_table(settings))
    styles.reset_index().to_parquet(settings.processed_dir / "player_styles.parquet", index=False)
    pairs = pair_scores(settings, stream, source, basis=basis)
    gold_sd = float(basis["raw"][basis["fit"]].std())
    del basis
    pairs.to_parquet(settings.processed_dir / "pair_synergy.parquet", index=False)
    grid = np.linspace(0.0, 1.0, 1001)
    combos = pairs.groupby(combination_keys(pairs["position_a"], pairs["position_b"]))["synergy"]
    names = sorted(combos.groups)
    teams = pairs.groupby(["match_id", "side"])["synergy"].sum().to_numpy()
    np.savez(
        settings.model_dir / "interaction_scores.npz",
        quantiles=np.quantile(pairs["synergy"].to_numpy(), grid),
        combos=np.array(names),
        combo_quantiles=np.stack([np.quantile(combos.get_group(name).to_numpy(), grid) for name in names]),
        team_quantiles=np.quantile(teams, grid),
        gold_sd=gold_sd,
    )
    return {
        "gold_sd": round(gold_sd, 1),
        "player_positions": int(len(styles)),
        "columns": len(columns),
        "pair_rows": int(len(pairs)),
        "synergy_sd": round(float(pairs["synergy"].std()), 6),
        "combinations": {name: round(float(combos.get_group(name).mean()), 6) for name in names},
    }


def combination_keys(left: pd.Series, right: pd.Series) -> np.ndarray:
    a, b = left.astype(str).to_numpy(), right.astype(str).to_numpy()
    return np.where(a < b, np.char.add(np.char.add(a, "+"), b), np.char.add(np.char.add(b, "+"), a))


def evidence_table(settings: Settings) -> pd.Series:
    from ..features.priority import EVIDENCE as PRIORITY_EVIDENCE, PRIORITY_COLUMNS
    from ..features.reaction import EVIDENCE as REACTION_EVIDENCE, REACTION_COLUMNS
    from ..features.tendency import EVIDENCE as TENDENCY_EVIDENCE, TENDENCY_COLUMNS

    parts = []
    for name, weight in ((REACTION_EVIDENCE, len(REACTION_COLUMNS)), (PRIORITY_EVIDENCE, len(PRIORITY_COLUMNS)), (TENDENCY_EVIDENCE, len(TENDENCY_COLUMNS))):
        path = settings.processed_dir / name
        if path.exists():
            parts.append((weight, pd.read_parquet(path)))
    if not parts:
        return pd.Series(dtype=float, name="evidence", index=pd.MultiIndex.from_arrays([[], []], names=KEY))
    total = sum(weight for weight, _ in parts)
    merged = None
    for weight, frame in parts:
        scaled = frame.set_index(KEY)["share"] * (weight / total)
        merged = scaled if merged is None else merged.add(scaled, fill_value=0.0)
    return merged.rename("evidence")
