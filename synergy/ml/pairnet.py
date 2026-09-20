import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import Settings, get_settings
from ..features.positions import POSITIONS
from ..ingest.premades import premade_pairs
from .interaction import SEED, _basis, evidence_table
from .mirrored import seat_games, seat_gold, seats_by_position

HIDDEN = 256
EPOCHS = 12
BATCH = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.1
VALIDATION = 0.1
PENALTIES = (1e2, 1e3, 1e4)
PLANTED_GOLD = 50.0
SEEDS = 5
FAMILY_SEEDS = 3
DEPTHS = (20, 50)
REFERENCE = 128
LEAST_SEATS = 8
LINEUPS = 20000
EVIDENCE_BANDS = (0.0, 0.02, 0.08, 0.2, 0.35)
LEAST_PAIRS = 500
GRID = np.linspace(0.0, 1.0, 1001)
REPORT_FILE = "pairnet_report.json"
MODEL_FILE = "pairnet.npz"
COMBINATIONS = [(a, b) for a in range(len(POSITIONS)) for b in range(a + 1, len(POSITIONS))]


class InteractionNet(nn.Module):
    def __init__(self, width: int, hidden: int = HIDDEN):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(3 * width, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, 1),
        )

    def side(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([first, second, first * second], dim=-1)).squeeze(-1)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return self.side(cells[:, 0], cells[:, 1]) - self.side(cells[:, 2], cells[:, 3])


def pair_index(blue: np.ndarray, red: np.ndarray, rows: np.ndarray) -> np.ndarray:
    out = []
    for first, second in COMBINATIONS:
        seats = (blue[rows, first], blue[rows, second], red[rows, first], red[rows, second])
        out.append(np.stack([column for seat in seats for column in (rows, seat)], axis=1))
    return np.concatenate(out).astype(np.int64)


def pair_target(gold: np.ndarray, index: np.ndarray) -> np.ndarray:
    picked = gold[index[:, 0::2], index[:, 1::2]]
    return picked[:, 0] + picked[:, 1] - picked[:, 2] - picked[:, 3]


def shuffle_partners(index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mixed = index.copy()
    per = len(index) // len(COMBINATIONS)
    for block in range(len(COMBINATIONS)):
        rows = slice(block * per, (block + 1) * per)
        order = rng.permutation(per)
        for column in (2, 3, 6, 7):
            mixed[rows, column] = index[rows, column][order]
    return mixed


def gather(reduced: torch.Tensor, index: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    cells = torch.stack([reduced[index[:, 2 * k], index[:, 2 * k + 1]] for k in range(4)], dim=1)
    return cells if mask is None else cells * mask


def additive(cells: torch.Tensor) -> torch.Tensor:
    return cells[:, 0] + cells[:, 1] - cells[:, 2] - cells[:, 3]


def explained(values: np.ndarray, guess: np.ndarray) -> float:
    return 1.0 - float(((values - guess) ** 2).sum()) / float(((values - values.mean()) ** 2).sum())


def batches(index: torch.Tensor, size: int = BATCH * 4):
    for start in range(0, len(index), size):
        yield index[start : start + size]


def linear_weights(reduced: torch.Tensor, index: torch.Tensor, target: torch.Tensor, check: torch.Tensor, check_target: torch.Tensor) -> tuple[torch.Tensor, float]:
    width = reduced.shape[-1]
    middle = float(target.mean())
    left = torch.zeros((width, width), dtype=torch.float64, device=reduced.device)
    right = torch.zeros(width, dtype=torch.float64, device=reduced.device)
    for start in range(0, len(index), BATCH * 4):
        design = additive(gather(reduced, index[start : start + BATCH * 4])).double()
        left += design.T @ design
        right += design.T @ (target[start : start + BATCH * 4].double() - middle)
    chosen = None
    for penalty in PENALTIES:
        weights = torch.linalg.solve(left + penalty * torch.eye(width, dtype=torch.float64, device=reduced.device), right).float()
        guess = torch.cat([additive(gather(reduced, rows)) @ weights for rows in batches(check)]) + middle
        score = 1.0 - float(((guess - check_target) ** 2).sum()) / float(((check_target - check_target.mean()) ** 2).sum())
        if chosen is None or score > chosen[0]:
            chosen = (score, weights)
    return chosen[1], middle


def linear_predict(reduced: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, middle: float) -> torch.Tensor:
    return torch.cat([additive(gather(reduced, rows)) @ weights for rows in batches(index)]) + middle


def train_interaction(
    reduced: torch.Tensor,
    index: torch.Tensor,
    residual: torch.Tensor,
    check: torch.Tensor,
    check_residual: torch.Tensor,
    held: torch.Tensor,
    seed: int,
    epochs: int = EPOCHS,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float], InteractionNet, float]:
    torch.manual_seed(seed)
    device = reduced.device
    scale = float(residual.std())
    model = InteractionNet(reduced.shape[-1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best, best_state, history = None, None, []
    order = torch.randperm(len(index), device=device)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(index), BATCH):
            rows = order[start : start + BATCH]
            guess = model(gather(reduced, index[rows], mask))
            loss = ((guess - residual[rows] / scale) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            guess = torch.cat([model(gather(reduced, rows, mask)) for rows in batches(check)]) * scale
            score = 1.0 - float(((guess - check_residual) ** 2).sum()) / float(((check_residual - check_residual.mean()) ** 2).sum())
        history.append(round(score, 5))
        if best is None or score > best:
            best, best_state = score, {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(gather(reduced, rows, mask)) for rows in batches(held)]) * scale, history, model, scale


def player_means(basis: dict, rows: np.ndarray) -> dict:
    reduced = basis["reduced"]
    dim = reduced.shape[-1]
    names = pd.Series(basis["seat_puuid"][rows].ravel()) + "|" + pd.Series(basis["seat_position"][rows].ravel())
    codes, keys = pd.factorize(names)
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) > 0])
    counts = np.diff(np.r_[starts, len(order)])
    sums = np.add.reduceat(reduced[rows].reshape(-1, dim)[order].astype(np.float64), starts, axis=0)
    return {
        "means": (sums / counts[:, None]).astype(np.float32),
        "positions": np.array([str(key).split("|")[1] for key in keys]),
        "counts": counts,
        "at": {str(key): index for index, key in enumerate(keys)},
    }


def with_evidence(table: dict, evidence: pd.Series) -> dict:
    known = {f"{puuid}|{position}": float(value) for (puuid, position), value in evidence.items() if pd.notna(value)}
    table["evidence"] = np.array([known.get(key, 0.0) for key in table["at"]], dtype=np.float32)
    return table


def reference_players(table: dict, rng: np.random.Generator, count: int = REFERENCE, least: int = LEAST_SEATS) -> np.ndarray:
    out = np.zeros((len(POSITIONS), count, table["means"].shape[-1]), dtype=np.float32)
    for index, position in enumerate(POSITIONS):
        pool = table["means"][(table["positions"] == position) & (table["counts"] >= least)]
        out[index] = pool[rng.choice(len(pool), size=count, replace=len(pool) < count)]
    return out


def corpus_pairs(basis: dict, rows: np.ndarray, blue: np.ndarray, red: np.ndarray, at: dict) -> dict:
    names = basis["seat_puuid"]
    combo, left, right, side = [], [], [], []
    for k, (i, j) in enumerate(COMBINATIONS):
        for s, seats in enumerate((blue, red)):
            a, b = seats[rows, i], seats[rows, j]
            left.append(np.array([at[f"{names[r, x]}|{POSITIONS[i]}"] for r, x in zip(rows, a)]))
            right.append(np.array([at[f"{names[r, x]}|{POSITIONS[j]}"] for r, x in zip(rows, b)]))
            combo.append(np.full(len(rows), k))
            side.append(np.arange(len(rows)) * 2 + s)
    return {
        "combo": np.concatenate(combo),
        "left": np.concatenate(left),
        "right": np.concatenate(right),
        "side": np.concatenate(side),
    }


def partner_means(models: list[tuple[InteractionNet, float]], table: dict, reference: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    means = torch.as_tensor(table["means"], device=device)
    refs = [torch.as_tensor(reference[k], device=device) for k in range(len(POSITIONS))]
    count, dim = reference.shape[1], reference.shape[2]
    out = np.zeros((len(means), len(POSITIONS)), dtype=np.float32)
    for i, own in enumerate(POSITIONS):
        rows = np.flatnonzero(table["positions"] == own)
        for j in range(len(POSITIONS)):
            if j == i:
                continue
            for start in range(0, len(rows), batch):
                chunk = rows[start : start + batch]
                mine = means[chunk][:, None, :].expand(-1, count, -1).reshape(-1, dim)
                theirs = refs[j][None, :, :].expand(len(chunk), -1, -1).reshape(-1, dim)
                found = ensemble_side(models, mine, theirs) if i < j else ensemble_side(models, theirs, mine)
                out[chunk, j] = found.reshape(len(chunk), count).mean(axis=1).cpu().numpy()
    return out


def corpus_fits(models: list[tuple[InteractionNet, float]], table: dict, corpus: dict, partner: np.ndarray, grands: np.ndarray, device: str, batch: int = 32768) -> np.ndarray:
    means = torch.as_tensor(table["means"], device=device)
    combo, left, right = corpus["combo"], corpus["left"], corpus["right"]
    own = np.zeros(len(combo), dtype=np.float32)
    for start in range(0, len(combo), batch):
        rows = slice(start, start + batch)
        own[rows] = ensemble_side(models, means[left[rows]], means[right[rows]]).cpu().numpy()
    first = np.array([a for a, _ in COMBINATIONS])[combo]
    second = np.array([b for _, b in COMBINATIONS])[combo]
    return own - partner[left, second] - partner[right, first] + grands[combo]


def ensemble_side(models: list[tuple[InteractionNet, float]], first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([model.side(first, second) * scale for model, scale in models]).mean(axis=0)


def centred_fit(models: list[tuple[InteractionNet, float]], left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, float, np.ndarray]:
    first = left[:, None, :].expand(-1, len(right), -1).reshape(-1, left.shape[-1])
    second = right[None, :, :].expand(len(left), -1, -1).reshape(-1, right.shape[-1])
    with torch.no_grad():
        grids = torch.stack([model.side(first, second) * scale for model, scale in models]).reshape(len(models), len(left), len(right))
    grid = grids.mean(axis=0)
    grand = float(grid.mean())
    return grid - grid.mean(axis=1, keepdim=True) - grid.mean(axis=0, keepdim=True) + grand, grand, grids.mean(axis=(1, 2)).cpu().numpy()


def family_masks(columns: list[str], device: str) -> dict[str, torch.Tensor]:
    families = sorted({name.split("_")[0] for name in columns})
    out = {}
    for family in families:
        keep = torch.tensor([0.0 if name.split("_")[0] == family else 1.0 for name in columns], device=device)
        out[family] = keep
    return out


def save_ensemble(
    path,
    models: list[tuple[InteractionNet, float]],
    basis: dict,
    columns: list[str],
    reference: np.ndarray,
    ridge_weights: np.ndarray,
    ridge_middle: float,
    device: str,
    lineups: int = LINEUPS,
    seed: int = SEED,
    table: dict | None = None,
    corpus: dict | None = None,
) -> dict:
    masks = family_masks(columns, device)
    families = sorted(masks)
    refs = [torch.as_tensor(reference[k], device=device) for k in range(len(POSITIONS))]
    combos, grands, grand_seeds, grids, grand_without = [], [], [], [], []
    for first, second in COMBINATIONS:
        fit, grand, by_seed = centred_fit(models, refs[first], refs[second])
        combos.append(f"{POSITIONS[first]}+{POSITIONS[second]}")
        grands.append(grand)
        grand_seeds.append(by_seed)
        grids.append(fit.cpu().numpy())
        grand_without.append([centred_fit(models, refs[first] * masks[f], refs[second] * masks[f])[1] for f in families])
    if table is not None and corpus is not None:
        partner = partner_means(models, table, reference, device)
        fits = corpus_fits(models, table, corpus, partner, np.array(grands), device)
        known = table["evidence"][corpus["left"]] * table["evidence"][corpus["right"]]
        band = np.searchsorted(np.array(EVIDENCE_BANDS), known, side="right") - 1
        quantiles = []
        for k in range(len(COMBINATIONS)):
            rows = []
            for b in range(len(EVIDENCE_BANDS)):
                found = fits[(corpus["combo"] == k) & (band == b)]
                if len(found) >= LEAST_PAIRS:
                    rows.append(np.quantile(found, GRID))
                else:
                    rows.append(rows[-1] if rows else np.quantile(fits[corpus["combo"] == k], GRID))
            quantiles.append(np.stack(rows))
        teams = np.bincount(corpus["side"], weights=fits)
    else:
        quantiles = [np.tile(np.quantile(grid.ravel(), GRID), (len(EVIDENCE_BANDS), 1)) for grid in grids]
        picks = np.random.default_rng(seed).integers(0, reference.shape[1], size=(lineups, len(POSITIONS)))
        teams = sum(grids[k][picks[:, a], picks[:, b]] for k, (a, b) in enumerate(COMBINATIONS))

    def stacked(layer: int, name: str) -> np.ndarray:
        return np.stack([getattr(model.body[layer], name).detach().cpu().numpy() for model, _ in models])

    np.savez(
        path,
        first_weight=stacked(0, "weight"),
        first_bias=stacked(0, "bias"),
        second_weight=stacked(3, "weight"),
        second_bias=stacked(3, "bias"),
        third_weight=stacked(6, "weight")[:, 0, :],
        third_bias=stacked(6, "bias")[:, 0],
        scale=np.array([scale for _, scale in models]),
        columns=np.array(columns),
        centre=basis["centre"],
        spread=basis["spread"],
        reference=reference,
        positions=np.array(list(POSITIONS)),
        combos=np.array(combos),
        grand=np.array(grands),
        grand_seeds=np.array(grand_seeds),
        families=np.array(families),
        grand_without=np.array(grand_without),
        fit_quantiles=np.stack(quantiles),
        evidence_bands=np.array(EVIDENCE_BANDS),
        team_quantiles=np.quantile(teams, GRID),
        ridge_weights=ridge_weights,
        ridge_middle=ridge_middle,
        units=np.array("gold at 15"),
    )
    return {
        name: {f"{EVIDENCE_BANDS[b]}+": round(float(found[b][900] - found[b][100]), 1) for b in (0, 2, len(EVIDENCE_BANDS) - 1)}
        for name, found in zip(combos, quantiles)
    }


def shape(values: np.ndarray) -> dict:
    centred = values - values.mean()
    spread = float(centred.std())
    standard = centred / spread if spread > 0 else centred
    return {
        "spread": round(spread, 1),
        "percentiles": {str(p): round(float(np.percentile(centred, p)), 1) for p in (1, 5, 25, 50, 75, 95, 99)},
        "widest": round(float(np.abs(centred).max()), 1),
        "skew": round(float((standard**3).mean()), 3),
        "kurtosis": round(float((standard**4).mean()), 3),
        "beyond_two_spreads": round(float((np.abs(standard) > 2).mean()), 4),
    }


def fit_pairnet(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    seed: int = SEED,
    source: str = "style",
    epochs: int = EPOCHS,
    seeds: int = SEEDS,
    family_seeds: int = FAMILY_SEEDS,
) -> dict:
    settings = settings or get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clock = time.time()
    basis = _basis(settings, stream, seed, source)
    columns = [str(name) for name in basis["columns"]]
    blue, red = seats_by_position(basis["seat_position"], basis["seat_side"])
    gold = seat_gold(settings, basis["match_id"], basis["seat_puuid"])
    games = seat_games(basis["seat_puuid"], basis["seat_position"])
    premade = premade_pairs(settings)
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    fit = np.array(sorted(set(basis["fit"]) & set(np.nonzero(sound)[0])))
    test = np.array(sorted(set(basis["test"]) & set(np.nonzero(sound)[0])))
    everyone = np.array(sorted(set(fit) | set(test)))
    table = with_evidence(player_means(basis, everyone), evidence_table(settings))
    reference = reference_players(table, np.random.default_rng(seed + 11))
    corpus = corpus_pairs(basis, everyone, blue, red, table["at"])
    reduced = torch.as_tensor(basis.pop("reduced"), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fit))
    cut = int(len(order) * (1.0 - VALIDATION))
    learn, check = fit[np.sort(order[:cut])], fit[np.sort(order[cut:])]
    raw = {name: pair_index(blue, red, rows) for name, rows in (("learn", learn), ("check", check), ("held", test))}
    index = {name: torch.as_tensor(rows, device=device) for name, rows in raw.items()}
    target = {name: torch.as_tensor(pair_target(gold, raw[name]), dtype=torch.float32, device=device) for name in index}
    held_values = target["held"].cpu().numpy()
    spread = float(target["learn"].std())
    print(
        f"pair network on {device}: {len(index['learn']):,} pair rows to learn, {len(index['check']):,} to check,"
        f" {len(index['held']):,} held out, spread {spread:,.0f} gold, {len(premade):,} premade pairs, basis ready in {time.time() - clock:.0f}s",
        flush=True,
    )

    weights, middle = linear_weights(reduced, index["learn"], target["learn"], index["check"], target["check"])
    linear = {name: linear_predict(reduced, index[name], weights, middle) for name in index}
    residual = {name: target[name] - linear[name] for name in index}
    ridge_held = linear["held"].cpu().numpy()
    alone = explained(held_values, ridge_held)
    report = {
        "device": device,
        "pair_rows": int(len(index["learn"]) + len(index["check"])),
        "held_out_rows": int(len(index["held"])),
        "target_spread": round(spread, 1),
        "epochs": epochs,
        "seeds": seeds,
        "cells_alone": round(alone, 5),
    }
    print(f"  linear ridge held out r2 {alone:+.5f}", flush=True)

    def interaction(name, learn_index, check_index, held_index, learn_residual, check_residual, held_truth, draw=0, mask=None, quiet=False):
        clock = time.time()
        guess, history, model, scale = train_interaction(
            reduced, learn_index, learn_residual, check_index, check_residual, held_index, seed + draw, epochs, mask
        )
        score = explained(held_truth, (linear["held"] + guess).cpu().numpy())
        gain = score - explained(held_truth, ridge_held)
        if not quiet:
            print(
                f"  {name:12} held out r2 {score:+.5f}, gain over linear {gain:+.5f}, spread {float(guess.std()):.1f} gold,"
                f" {time.time() - clock:.0f}s, checks {history}",
                flush=True,
            )
        return {"held_out": round(score, 5), "gain": round(gain, 5), "spread": round(float(guess.std()), 1), "checks": history}, guess, (model, scale)

    def true_draw(draw):
        return index["learn"], index["check"], index["held"], residual["learn"], residual["check"], held_values

    def shuffled_draw(draw):
        mixer = np.random.default_rng(seed + 100 + draw)
        mixed = {name: torch.as_tensor(shuffle_partners(raw[name], mixer), device=device) for name in raw}
        return mixed["learn"], mixed["check"], mixed["held"], residual["learn"], residual["check"], held_values

    def repeated(name, draws, mask=None, draws_wanted=None, quiet=False, models=None):
        runs, kept = [], []
        for draw in range(seeds if draws_wanted is None else draws_wanted):
            run, guess, fitted = interaction(f"{name} {draw}", *draws(draw), draw=draw, mask=mask, quiet=quiet)
            runs.append(run)
            kept.append(guess)
            if models is not None:
                models.append(fitted)
        gains = np.array([run["gain"] for run in runs])
        return {
            "gain": round(float(gains.mean()), 5),
            "gain_spread": round(float(gains.std(ddof=1)) if len(gains) > 1 else 0.0, 5),
            "gains": [run["gain"] for run in runs],
            "held_out": round(float(np.mean([run["held_out"] for run in runs])), 5),
            "spread": round(float(np.mean([run["spread"] for run in runs])), 1),
        }, kept

    models = []
    report["network"], network_guesses = repeated("network", true_draw, models=models)
    report["shuffled"], shuffled_guesses = repeated("shuffled", shuffled_draw)
    difference = report["network"]["gain"] - report["shuffled"]["gain"]
    error = float(np.sqrt((report["network"]["gain_spread"] ** 2 + report["shuffled"]["gain_spread"] ** 2) / max(seeds, 1)))
    report["pairing"] = {"gain": round(difference, 5), "error": round(error, 5), "gold": round(float(np.sqrt(max(difference, 0.0))) * spread, 1)}
    print(f"  pairing beyond shuffled partners: gain {difference:+.5f} ± {error:.5f}, about {report['pairing']['gold']:.0f} gold", flush=True)

    torch.manual_seed(seed + 7)
    hidden = InteractionNet(reduced.shape[-1], hidden=64).to(device)
    hidden.eval()
    with torch.no_grad():
        planted = {name: torch.cat([hidden(gather(reduced, rows)) for rows in batches(index[name])]) for name in index}
    scale = PLANTED_GOLD / float(planted["learn"].std())
    planted_truth = held_values + (planted["held"] * scale).cpu().numpy()
    found, _, _ = interaction(
        "planted", index["learn"], index["check"], index["held"],
        residual["learn"] + planted["learn"] * scale, residual["check"] + planted["check"] * scale, planted_truth,
    )
    report["planted"] = {f"{PLANTED_GOLD:.0f} gold": found["gain"], "expected": round((PLANTED_GOLD / spread) ** 2 + report["network"]["gain"], 5)}
    report["informative"] = bool(difference > 2.0 * error and found["gain"] > report["network"]["gain"])

    guesses = np.stack([one.cpu().numpy() for one in network_guesses])
    guess = guesses.mean(axis=0)
    report["ensemble"] = {
        "held_out": round(explained(held_values, ridge_held + guess), 5),
        "gain": round(explained(held_values, ridge_held + guess) - alone, 5),
        "spread": round(float(guess.std()), 1),
        "agreement": round(float(np.corrcoef(guesses)[np.triu_indices(len(guesses), 1)].mean()), 3) if len(guesses) > 1 else 1.0,
    }
    print(
        f"  seeds averaged: gain {report['ensemble']['gain']:+.5f}, spread {report['ensemble']['spread']:.1f} gold,"
        f" seeds agree {report['ensemble']['agreement']:.3f}",
        flush=True,
    )
    report["tails"] = shape(guess)
    report["tails_shuffled"] = shape(np.stack([one.cpu().numpy() for one in shuffled_guesses]).mean(axis=0))
    print(f"  tails of the fitted pairing: {report['tails']}", flush=True)
    print(f"  tails with partners shuffled: {report['tails_shuffled']}", flush=True)

    per = len(raw["held"]) // len(COMBINATIONS)
    report["combinations"] = {}
    for block, (first, second) in enumerate(COMBINATIONS):
        rows = slice(block * per, (block + 1) * per)
        name = f"{POSITIONS[first]}+{POSITIONS[second]}"
        gain = explained(held_values[rows], ridge_held[rows] + guess[rows]) - explained(held_values[rows], ridge_held[rows])
        each = [explained(held_values[rows], ridge_held[rows] + one[rows]) - explained(held_values[rows], ridge_held[rows]) for one in guesses]
        report["combinations"][name] = {
            "gain": round(gain, 5),
            "seed_spread": round(float(np.std(each, ddof=1)) if len(each) > 1 else 0.0, 5),
            "spread": round(float(guess[rows].std()), 1),
        }
    print(f"  by combination: {report['combinations']}", flush=True)

    held_raw = raw["held"]
    depth = np.minimum.reduce([games[held_raw[:, 2 * k], held_raw[:, 2 * k + 1]] for k in range(4)])
    names = basis["seat_puuid"]
    ours = np.array([tuple(sorted((names[row[0], row[1]], names[row[2], row[3]]))) in premade for row in held_raw])
    report["slices"] = {}
    for name, mask in (
        *((f"both pairs {d}+ games in position", depth >= d) for d in DEPTHS),
        ("premade", ours),
        ("strangers", ~ours),
    ):
        if int(mask.sum()) < 500:
            continue
        gain = explained(held_values[mask], ridge_held[mask] + guess[mask]) - explained(held_values[mask], ridge_held[mask])
        report["slices"][name] = {"rows": int(mask.sum()), "gain": round(gain, 5), "spread": round(float(guess[mask].std()), 1)}
    print(f"  by slice: {report['slices']}", flush=True)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    report["served_fit_spread"] = save_ensemble(
        settings.model_dir / MODEL_FILE, models, basis, columns, reference, weights.cpu().numpy(), middle, device,
        table=table, corpus=corpus,
    )
    print(
        f"  saved {len(models)} networks in {time.time() - clock:.0f}s, fit between the 10th and 90th percentile"
        f" over {len(corpus['combo']):,} corpus pairs by combination: {report['served_fit_spread']}",
        flush=True,
    )

    report["families"] = {}
    for family, mask in family_masks(columns, device).items() if family_seeds > 0 else []:
        kept, _ = repeated(f"without {family}", true_draw, mask=mask, draws_wanted=family_seeds, quiet=True)
        null, _ = repeated(f"without {family}, shuffled", shuffled_draw, mask=mask, draws_wanted=family_seeds, quiet=True)
        pairing = kept["gain"] - null["gain"]
        error = float(np.sqrt((kept["gain_spread"] ** 2 + null["gain_spread"] ** 2) / max(family_seeds, 1)))
        report["families"][family] = {
            "columns": int((mask == 0).sum().item()),
            "pairing_without": round(pairing, 5),
            "error": round(error, 5),
            "lost": round(difference - pairing, 5),
        }
        print(
            f"  without {family:6} ({report['families'][family]['columns']:3} cells): pairing {pairing:+.5f} ± {error:.5f},"
            f" lost {report['families'][family]['lost']:+.5f}",
            flush=True,
        )

    (settings.model_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
