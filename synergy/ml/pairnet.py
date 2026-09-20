import json
import time

import numpy as np
import torch
from torch import nn

from ..config import Settings, get_settings
from ..features.positions import POSITIONS
from .interaction import SEED, _basis
from .mirrored import seat_gold, seats_by_position

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
REPORT_FILE = "pairnet_report.json"
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


def train_interaction(
    reduced: torch.Tensor,
    index: torch.Tensor,
    residual: torch.Tensor,
    check: torch.Tensor,
    check_residual: torch.Tensor,
    held: torch.Tensor,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[torch.Tensor, list[float]]:
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
    sound = (blue >= 0).all(axis=1) & (red >= 0).all(axis=1) & np.isfinite(gold).all(axis=1)
    fit = np.array(sorted(set(basis["fit"]) & set(np.nonzero(sound)[0])))
    test = np.array(sorted(set(basis["test"]) & set(np.nonzero(sound)[0])))
    reduced = torch.as_tensor(basis.pop("reduced"), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fit))
    cut = int(len(order) * (1.0 - VALIDATION))
    learn, check = fit[np.sort(order[:cut])], fit[np.sort(order[cut:])]
    index = {name: torch.as_tensor(pair_index(blue, red, rows), device=device) for name, rows in (("learn", learn), ("check", check), ("held", test))}
    target = {name: torch.as_tensor(pair_target(gold, index[name].cpu().numpy()), dtype=torch.float32, device=device) for name in index}
    held_values = target["held"].cpu().numpy()
    spread = float(target["learn"].std())
    print(
        f"pair network on {device}: {len(index['learn']):,} pair rows to learn, {len(index['check']):,} to check,"
        f" {len(index['held']):,} held out, spread {spread:,.0f} gold, basis ready in {time.time() - clock:.0f}s",
        flush=True,
    )

    weights, middle = linear_weights(reduced, index["learn"], target["learn"], index["check"], target["check"])
    linear = {name: linear_predict(reduced, index[name], weights, middle) for name in index}
    residual = {name: target[name] - linear[name] for name in index}
    report = {
        "device": device,
        "pair_rows": int(len(index["learn"]) + len(index["check"])),
        "held_out_rows": int(len(index["held"])),
        "target_spread": round(spread, 1),
        "epochs": epochs,
        "cells_alone": round(explained(held_values, linear["held"].cpu().numpy()), 5),
    }
    print(f"  linear ridge held out r2 {report['cells_alone']:+.5f}", flush=True)

    def interaction(name, learn_index, check_index, held_index, learn_residual, check_residual, held_truth, draw=0):
        clock = time.time()
        guess, history = train_interaction(reduced, learn_index, learn_residual, check_index, check_residual, held_index, seed + draw, epochs)
        score = explained(held_truth, (linear["held"] + guess).cpu().numpy())
        gain = score - explained(held_truth, linear["held"].cpu().numpy())
        print(f"  {name:10} held out r2 {score:+.5f}, gain over linear {gain:+.5f}, spread {float(guess.std()):.1f} gold, {time.time() - clock:.0f}s, checks {history}", flush=True)
        return {"held_out": round(score, 5), "gain": round(gain, 5), "spread": round(float(guess.std()), 1), "checks": history}

    def repeated(name, draws):
        runs = [interaction(f"{name} {draw}", *draws(draw), draw=draw) for draw in range(seeds)]
        gains = np.array([run["gain"] for run in runs])
        return {
            "gain": round(float(gains.mean()), 5),
            "gain_spread": round(float(gains.std(ddof=1)) if seeds > 1 else 0.0, 5),
            "gains": [run["gain"] for run in runs],
            "held_out": round(float(np.mean([run["held_out"] for run in runs])), 5),
            "spread": round(float(np.mean([run["spread"] for run in runs])), 1),
        }

    report["network"] = repeated(
        "network", lambda draw: (index["learn"], index["check"], index["held"], residual["learn"], residual["check"], held_values)
    )

    def shuffled_draw(draw):
        mixer = np.random.default_rng(seed + 100 + draw)
        mixed = {name: torch.as_tensor(shuffle_partners(index[name].cpu().numpy(), mixer), device=device) for name in index}
        return mixed["learn"], mixed["check"], mixed["held"], residual["learn"], residual["check"], held_values

    report["shuffled"] = repeated("shuffled", shuffled_draw)
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
    found = interaction(
        "planted", index["learn"], index["check"], index["held"],
        residual["learn"] + planted["learn"] * scale, residual["check"] + planted["check"] * scale, planted_truth,
    )
    report["planted"] = {f"{PLANTED_GOLD:.0f} gold": found["gain"], "expected": round((PLANTED_GOLD / spread) ** 2 + report["network"]["gain"], 5)}
    report["informative"] = bool(difference > 2.0 * error and found["gain"] > report["network"]["gain"])

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    (settings.model_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
