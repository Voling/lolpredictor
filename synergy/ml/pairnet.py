import json
import time

import numpy as np
import torch
from torch import nn

from ..config import Settings, get_settings
from ..features.positions import POSITIONS
from ..ingest.premades import premade_pairs
from .interaction import SEED, _basis
from .mirrored import SEATS_FILE, seat_games, seat_gold, seats_by_position

HIDDEN = 256
EPOCHS = 12
BATCH = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.1
VALIDATION = 0.1
PENALTIES = (1e2, 1e3, 1e4)
PLANTED_GOLD = 100.0
SEEDS = 5
DEPTHS = (20, 50)
REPORT_FILE = "pairnet_report.json"
COMBINATIONS = [(a, b) for a in range(len(POSITIONS)) for b in range(a + 1, len(POSITIONS))]


def _body(width: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, hidden),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(hidden, 1),
    )


class InteractionNet(nn.Module):
    def __init__(self, width: int, hidden: int = HIDDEN):
        super().__init__()
        self.body = _body(3 * width, hidden)

    def side(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([first, second, first * second], dim=-1)).squeeze(-1)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return self.side(cells[:, 0], cells[:, 1]) - self.side(cells[:, 2], cells[:, 3])


class CurvedNet(nn.Module):
    def __init__(self, width: int, hidden: int = HIDDEN):
        super().__init__()
        self.body = _body(width, hidden)

    def side(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (self.body(first) + self.body(second)).squeeze(-1)

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


def gather(reduced: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.stack([reduced[index[:, 2 * k], index[:, 2 * k + 1]] for k in range(4)], dim=1)


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


def seat_ridge_predict(reduced: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, combo: torch.Tensor, middle: float) -> torch.Tensor:
    first = torch.as_tensor([a for a, _ in COMBINATIONS], device=reduced.device)[combo]
    second = torch.as_tensor([b for _, b in COMBINATIONS], device=reduced.device)[combo]
    out = []
    for start in range(0, len(index), BATCH * 4):
        stop = start + BATCH * 4
        cells = gather(reduced, index[start:stop])
        own = ((cells[:, 0] - cells[:, 2]) * weights[first[start:stop]]).sum(dim=1)
        partner = ((cells[:, 1] - cells[:, 3]) * weights[second[start:stop]]).sum(dim=1)
        out.append(own + partner)
    return torch.cat(out) + middle


def train_interaction(
    reduced: torch.Tensor,
    index: torch.Tensor,
    residual: torch.Tensor,
    check: torch.Tensor,
    check_residual: torch.Tensor,
    held: torch.Tensor,
    seed: int,
    epochs: int = EPOCHS,
    build=InteractionNet,
) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    device = reduced.device
    scale = float(residual.std())
    model = build(reduced.shape[-1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best, best_state, history = None, None, []
    order = torch.randperm(len(index), device=device)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(index), BATCH):
            rows = order[start : start + BATCH]
            guess = model(gather(reduced, index[rows]))
            loss = ((guess - residual[rows] / scale) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            guess = torch.cat([model(gather(reduced, rows)) for rows in batches(check)]) * scale
            score = 1.0 - float(((guess - check_residual) ** 2).sum()) / float(((check_residual - check_residual.mean()) ** 2).sum())
        history.append(round(score, 5))
        if best is None or score > best:
            best, best_state = score, {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(gather(reduced, rows)) for rows in batches(held)]) * scale, history


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
) -> dict:
    settings = settings or get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clock = time.time()
    basis = _basis(settings, stream, seed, source)
    blue, red = seats_by_position(basis["seat_position"], basis["seat_side"])
    gold = seat_gold(settings, basis["match_id"], basis["seat_puuid"])
    games = seat_games(basis["seat_puuid"], basis["seat_position"])
    premade = premade_pairs(settings)
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    fit = np.array(sorted(set(basis["fit"]) & set(np.nonzero(sound)[0])))
    test = np.array(sorted(set(basis["test"]) & set(np.nonzero(sound)[0])))
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

    shared_weights, shared_middle = linear_weights(reduced, index["learn"], target["learn"], index["check"], target["check"])
    shared = explained(held_values, linear_predict(reduced, index["held"], shared_weights, shared_middle).cpu().numpy())
    with np.load(settings.model_dir / SEATS_FILE, allow_pickle=True) as saved:
        if [str(name) for name in saved["columns"]] != [str(name) for name in basis["columns"]]:
            raise ValueError("seat_weights.npz was fitted on different columns, run mirrored --positions-only first")
        seat_weights = torch.as_tensor(saved["weights"], dtype=torch.float32, device=device)
    combo = {name: torch.as_tensor(np.repeat(np.arange(len(COMBINATIONS)), len(raw[name]) // len(COMBINATIONS)), device=device) for name in raw}
    middle = float((target["learn"] - seat_ridge_predict(reduced, index["learn"], seat_weights, combo["learn"], 0.0)).mean())
    linear = {name: seat_ridge_predict(reduced, index[name], seat_weights, combo[name], middle) for name in index}
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
        "shared_ridge": round(shared, 5),
        "cells_alone": round(alone, 5),
    }
    print(f"  one ridge for every position held out r2 {shared:+.5f}, the seat ridge per position {alone:+.5f}", flush=True)

    def run(name, build, learn_residual, check_residual, held_truth, draw):
        clock = time.time()
        guess, history = train_interaction(
            reduced, index["learn"], learn_residual, index["check"], check_residual, index["held"], seed + draw, epochs, build
        )
        score = explained(held_truth, (linear["held"] + guess).cpu().numpy())
        gain = score - explained(held_truth, ridge_held)
        print(
            f"  {name:11} {draw} held out r2 {score:+.5f}, gain over linear {gain:+.5f}, spread {float(guess.std()):.1f} gold,"
            f" {time.time() - clock:.0f}s, checks {history}",
            flush=True,
        )
        return gain, guess.cpu().numpy()

    def repeated(name, build):
        found = [run(name, build, residual["learn"], residual["check"], held_values, draw) for draw in range(seeds)]
        gains = np.array([gain for gain, _ in found])
        guesses = np.stack([guess for _, guess in found])
        return {
            "gain": round(float(gains.mean()), 5),
            "gain_spread": round(float(gains.std(ddof=1)) if seeds > 1 else 0.0, 5),
            "gains": [round(float(gain), 5) for gain in gains],
            "spread": round(float(guesses.std(axis=1).mean()), 1),
        }, guesses

    report["network"], network_guesses = repeated("network", InteractionNet)
    report["curved"], curved_guesses = repeated("curved", CurvedNet)
    difference = report["network"]["gain"] - report["curved"]["gain"]
    error = float(np.sqrt((report["network"]["gain_spread"] ** 2 + report["curved"]["gain_spread"] ** 2) / max(seeds, 1)))
    report["interaction"] = {"gain": round(difference, 5), "error": round(error, 5), "gold": round(float(np.sqrt(max(difference, 0.0))) * spread, 1)}
    print(
        f"  interaction beyond curved single player effects: gain {difference:+.5f} ± {error:.5f}, about {report['interaction']['gold']:.0f} gold",
        flush=True,
    )

    torch.manual_seed(seed + 7)
    hidden = InteractionNet(reduced.shape[-1], hidden=64).to(device)
    hidden.eval()
    with torch.no_grad():
        planted = {name: torch.cat([hidden(gather(reduced, rows)) for rows in batches(index[name])]) for name in index}
    scale = PLANTED_GOLD / float(planted["learn"].std())
    planted_truth = held_values + (planted["held"] * scale).cpu().numpy()
    planted_learn, planted_check = residual["learn"] + planted["learn"] * scale, residual["check"] + planted["check"] * scale
    planted_network, _ = run("planted net", InteractionNet, planted_learn, planted_check, planted_truth, 0)
    planted_curved, _ = run("planted crv", CurvedNet, planted_learn, planted_check, planted_truth, 0)
    recovered = (planted_network - planted_curved) - (report["network"]["gains"][0] - report["curved"]["gains"][0])
    report["planted"] = {
        f"{PLANTED_GOLD:.0f} gold": round(recovered, 5),
        "single_player_part": round(planted_curved - report["curved"]["gains"][0], 5),
        "expected_total": round((PLANTED_GOLD / spread) ** 2, 5),
    }
    print(
        f"  planted {PLANTED_GOLD:.0f} gold: interaction recovered {recovered:+.5f}, single player part {report['planted']['single_player_part']:+.5f}",
        flush=True,
    )
    report["informative"] = bool(difference > 2.0 * error and recovered > 0.0)

    network_guess, curved_guess = network_guesses.mean(axis=0), curved_guesses.mean(axis=0)
    interaction_guess = network_guess - curved_guess
    with_network, with_curved = explained(held_values, ridge_held + network_guess), explained(held_values, ridge_held + curved_guess)
    report["ensemble"] = {
        "network_gain": round(with_network - alone, 5),
        "curved_gain": round(with_curved - alone, 5),
        "interaction_gain": round(with_network - with_curved, 5),
        "interaction_spread": round(float(interaction_guess.std()), 1),
    }
    print(f"  seeds averaged: {report['ensemble']}", flush=True)
    report["tails"] = shape(interaction_guess)
    report["tails_curved"] = shape(curved_guess)

    per = len(raw["held"]) // len(COMBINATIONS)
    report["combinations"] = {}
    for block, (first, second) in enumerate(COMBINATIONS):
        rows = slice(block * per, (block + 1) * per)
        name = f"{POSITIONS[first]}+{POSITIONS[second]}"
        with_network = explained(held_values[rows], ridge_held[rows] + network_guess[rows])
        with_curved = explained(held_values[rows], ridge_held[rows] + curved_guess[rows])
        each = [
            explained(held_values[rows], ridge_held[rows] + net[rows]) - explained(held_values[rows], ridge_held[rows] + crv[rows])
            for net, crv in zip(network_guesses, curved_guesses)
        ]
        report["combinations"][name] = {
            "interaction_gain": round(with_network - with_curved, 5),
            "seed_spread": round(float(np.std(each, ddof=1)) if len(each) > 1 else 0.0, 5),
            "curved_gain": round(with_curved - explained(held_values[rows], ridge_held[rows]), 5),
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
        with_network = explained(held_values[mask], ridge_held[mask] + network_guess[mask])
        with_curved = explained(held_values[mask], ridge_held[mask] + curved_guess[mask])
        report["slices"][name] = {
            "rows": int(mask.sum()),
            "interaction_gain": round(with_network - with_curved, 5),
            "curved_gain": round(with_curved - explained(held_values[mask], ridge_held[mask]), 5),
        }
    print(f"  by slice: {report['slices']}", flush=True)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    (settings.model_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
