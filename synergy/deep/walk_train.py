import json
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from ..config import Settings, get_settings
from ..ml.gold import team_advantage
from .walk import MatchWalk, batches

BATCH = 128
EPOCHS = 10
LR = 3e-4
PATIENCE = 3
HOLDOUT = 0.2
SEED = 0
TARGET = "advantage"
CHANNELS = ("lead", "kind", "actor", "victim", "region", "seat_region", "clock")


def _predict(model, data, index, batch, device):
    model.eval()
    guesses, truth = [], []
    with torch.no_grad():
        for *inputs, target in batches(data, index, batch, device, shuffle=False, target=TARGET):
            guesses.append(model(*inputs).float().cpu().numpy())
            truth.append(target.cpu().numpy())
    return np.concatenate(guesses), np.concatenate(truth)


def _r2(actual: np.ndarray, guess: np.ndarray) -> float:
    return 1.0 - float(((guess - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())


def _last_lead(data: dict) -> np.ndarray:
    mask = data["mask"]
    final = np.clip(mask.sum(axis=1) - 1, 0, None)
    return data["lead"][np.arange(len(final)), final].astype(float)


def train_walk(
    settings: Settings | None = None,
    stream: str = "stream.npz",
    epochs: int = EPOCHS,
    batch: int = BATCH,
    use_lead: bool = False,
) -> dict:
    settings = settings or get_settings()
    raw = np.load(settings.processed_dir / stream, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    gold = team_advantage(settings).reindex(data["match_id"])
    keep = gold.notna().to_numpy()
    per_match = [key for key, value in data.items() if np.ndim(value) and len(value) == len(keep)]
    for key in per_match:
        data[key] = data[key][keep]
    total = len(data["match_id"])
    order = np.random.default_rng(SEED).permutation(total)
    cut = int(total * (1.0 - HOLDOUT))
    train, test = order[:cut], order[cut:]
    advantage = gold.to_numpy(dtype=float)[keep]
    scale = float(advantage[train].std())
    data[TARGET] = ((advantage - advantage[train].mean()) / scale).astype(np.float32)
    lead = _last_lead(data)
    if not use_lead:
        data["lead"] = np.zeros_like(data["lead"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = MatchWalk(
        kinds=int(len(data["kinds"])),
        regions=int(len(data["regions"])),
        champions=int(len(data["champions"])),
        span=int(data["kind"].shape[1]),
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    fast = device == "cuda" and torch.cuda.is_bf16_supported()
    loss_fn = nn.MSELoss()
    history, best, stale = [], None, 0

    for epoch in range(epochs):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for *inputs, target in batches(data, train, batch, device, target=TARGET):
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=fast):
                loss = loss_fn(model(*inputs).float(), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
        guess, actual = _predict(model, data, test, batch, device)
        held = float(((guess - actual) ** 2).mean())
        history.append(
            {
                "epoch": epoch,
                "train_mse": round(running / seen, 5),
                "mse": round(held, 5),
                "r2": round(_r2(actual, guess), 5),
                "seconds": round(time.time() - started),
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best is None or held < best["mse"]:
            best, stale = history[-1], 0
            settings.model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), settings.model_dir / "match_walk.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(torch.load(settings.model_dir / "match_walk.pt", map_location=device))
    guess, actual = _predict(model, data, test, batch, device)
    board = LinearRegression().fit(lead[train].reshape(-1, 1), data[TARGET][train])
    plain = board.predict(lead[test].reshape(-1, 1))
    walked, simple = _r2(actual, guess), _r2(actual, plain)

    ablation = {}
    shuffle = np.random.default_rng(SEED).permutation(len(test))
    for channel in CHANNELS:
        swapped = dict(data)
        swapped[channel] = data[channel].copy()
        swapped[channel][test] = data[channel][test][shuffle]
        spoiled, _ = _predict(model, swapped, test, batch, device)
        ablation[channel] = {"r2": round(_r2(actual, spoiled), 5)}

    report = {
        "target": "gold plus objectives at 15, blue minus red, standardised on the training rows",
        "lead_channel": "fed to the model" if use_lead else "zeroed, the walk sees actions only",
        "gold_sd": round(scale, 1),
        "matches": total,
        "train": len(train),
        "held_out": len(test),
        "device": device,
        "bfloat16": fast,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best": best,
        "r2": round(walked, 5),
        "rmse_gold": round(float(np.sqrt(((guess - actual) ** 2).mean())) * scale, 1),
        "baselines": {"gold_lead_at_last_event_r2": round(simple, 5)},
        "skill_over_gold_lead": round(1.0 - (1.0 - walked) / (1.0 - simple), 4),
        "ablation": ablation,
        "history": history,
    }
    with open(settings.model_dir / "walk_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
