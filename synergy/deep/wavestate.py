import json
import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ..config import Settings, get_settings
from .lanes import ANCHORS, LANES, MAX_MINUTES, OBSERVED, load_lane_sequences

logger = logging.getLogger(__name__)

RECONSTRUCTED = ["blue_progress", "red_progress", "blue_cs_delta", "red_cs_delta"]
RECONSTRUCTED_INDEX = [OBSERVED.index(name) for name in RECONSTRUCTED]
PRESENCE_INDEX = [OBSERVED.index(name) for name in ("blue_in_lane", "red_in_lane", "blue_in_lane", "red_in_lane")]
PRIOR_MEAN = 1.0
PRIOR_SD = 0.4
KL_WEIGHT = 0.01


class WaveState(nn.Module):
    def __init__(self, observed: int, lanes: int, hidden: int = 64):
        super().__init__()
        self.lane_embedding = nn.Embedding(lanes, 8)
        self.encoder = nn.GRU(observed + 8, hidden, batch_first=True, bidirectional=True)
        self.posterior = nn.Linear(hidden * 2, 2)
        self.transition = nn.Sequential(nn.Linear(1 + 8 + 1, 32), nn.Tanh(), nn.Linear(32, 1))
        self.standoff = nn.Sequential(nn.Linear(8 + 1, 16), nn.Tanh(), nn.Linear(16, 2))
        self.farm = nn.Sequential(nn.Linear(1 + 8 + 1, 24), nn.Tanh(), nn.Linear(24, 2))
        self.anchor_slope = nn.Parameter(torch.tensor(1.0))
        self.anchor_bias = nn.Parameter(torch.zeros(len(ANCHORS)))
        self.log_noise = nn.Parameter(torch.zeros(len(RECONSTRUCTED)))
        self.log_transition_sd = nn.Parameter(torch.tensor(-1.5))

    def forward(self, observed, lane):
        batch, steps, _ = observed.shape
        lane_vector = self.lane_embedding(lane)[:, None].expand(batch, steps, -1)
        encoded, _ = self.encoder(torch.cat([observed, lane_vector], dim=-1))
        stats = self.posterior(encoded)
        mean = stats[..., :1] + PRIOR_MEAN
        log_var = stats[..., 1:].clamp(-6.0, 2.0)
        state = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var) if self.training else mean
        clock = observed[..., OBSERVED.index("clock"), None]
        offsets = nn.functional.softplus(self.standoff(torch.cat([lane_vector, clock], dim=-1)))
        flank = torch.cat([state - offsets[..., :1], state + offsets[..., 1:]], dim=-1)
        farmed = self.farm(torch.cat([state, lane_vector, clock], dim=-1))
        reconstruction = torch.cat([flank, farmed], dim=-1)
        slope = nn.functional.softplus(self.anchor_slope)
        centred = state - PRIOR_MEAN
        anchor_logits = torch.cat([centred * slope, -centred * slope], dim=-1) + self.anchor_bias
        return state, mean, log_var, reconstruction, anchor_logits, lane_vector, clock

    def predicted_next(self, state, lane_vector, clock):
        return state + self.transition(torch.cat([state, lane_vector, clock], dim=-1))


def _loss(model, batch, anchor_weight):
    observed, anchors, mask, lane = batch
    state, mean, log_var, reconstruction, anchor_logits, lane_vector, clock = model(observed, lane)
    target = observed[..., RECONSTRUCTED_INDEX]
    presence = observed[..., PRESENCE_INDEX]
    weight = presence * mask[..., None]
    noise = torch.exp(model.log_noise).clamp_min(1e-3)
    reconstruction_loss = (
        (((reconstruction - target) ** 2) / (2 * noise**2) + torch.log(noise)) * weight
    ).sum() / weight.sum().clamp_min(1.0)

    prior_mean = torch.cat(
        [
            torch.full_like(mean[:, :1], PRIOR_MEAN),
            model.predicted_next(state[:, :-1], lane_vector[:, :-1], clock[:, :-1]),
        ],
        dim=1,
    )
    prior_sd = torch.cat(
        [
            torch.full_like(mean[:, :1], PRIOR_SD),
            torch.exp(model.log_transition_sd).expand_as(mean[:, 1:]),
        ],
        dim=1,
    ).clamp_min(1e-2)
    posterior_sd = torch.exp(0.5 * log_var)
    kl = (
        torch.log(prior_sd / posterior_sd)
        + (posterior_sd**2 + (mean - prior_mean) ** 2) / (2 * prior_sd**2)
        - 0.5
    )
    kl_loss = (kl * mask[..., None]).sum() / mask.sum().clamp_min(1.0)

    anchor_loss = torch.tensor(0.0, device=observed.device)
    if anchor_weight > 0:
        bce = nn.functional.binary_cross_entropy_with_logits(anchor_logits, anchors, reduction="none")
        anchor_loss = (bce * mask[..., None]).sum() / mask.sum().clamp_min(1.0)
    return reconstruction_loss + KL_WEIGHT * kl_loss + anchor_weight * anchor_loss, mean


def train_wave_state(
    settings: Settings | None = None,
    epochs: int = 120,
    anchor_weight: float = 0.0,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    settings = settings or get_settings()
    data = load_lane_sequences(settings)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    matches = np.array(sorted(set(data["match_id"])))
    held = set(rng.choice(matches, size=max(len(matches) // 5, 1), replace=False))
    test = np.array([mid in held for mid in data["match_id"]])

    observed = torch.from_numpy(data["observed"]).float().to(device)
    anchors = torch.from_numpy(data["anchors"]).float().to(device)
    mask = torch.from_numpy(data["mask"]).float().to(device)
    lane = torch.from_numpy(data["lane"].astype(np.int64)).to(device)

    model = WaveState(observed.shape[-1], len(LANES)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_index = np.flatnonzero(~test)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_index)
        losses = []
        for start in range(0, len(train_index), batch_size):
            chunk = torch.from_numpy(train_index[start : start + batch_size]).to(device)
            loss, _ = _loss(
                model, (observed[chunk], anchors[chunk], mask[chunk], lane[chunk]), anchor_weight
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            losses.append(float(loss.item()))
        if epoch % 30 == 0:
            logger.info("epoch %s loss %.4f", epoch, float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        _, latent, _, _, _, _, _ = model(observed, lane)
    latent = latent.squeeze(-1).cpu().numpy()

    flat_mask = data["mask"]
    blue_plate = data["anchors"][..., 0][flat_mask]
    red_plate = data["anchors"][..., 1][flat_mask]
    values = latent[flat_mask]
    held_rows = np.repeat(test[:, None], MAX_MINUTES, axis=1)[flat_mask]

    report = {
        "lanes": int(len(data["mask"])),
        "lane_minutes": int(flat_mask.sum()),
        "held_out_matches": int(len(held)),
        "anchor_weight": anchor_weight,
        "latent_mean": round(float(values.mean()), 3),
        "latent_sd": round(float(values.std()), 3),
        "blue_plate_auc_heldout": round(float(roc_auc_score(blue_plate[held_rows], values[held_rows])), 4),
        "red_plate_auc_heldout": round(
            float(roc_auc_score(red_plate[held_rows], -values[held_rows])), 4
        ),
        "latent_when_blue_plate": round(float(values[blue_plate == 1].mean()), 3),
        "latent_when_red_plate": round(float(values[red_plate == 1].mean()), 3),
        "latent_when_quiet": round(float(values[(blue_plate == 0) & (red_plate == 0)].mean()), 3),
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, settings.model_dir / "wave_state.pt")
    np.savez_compressed(
        settings.processed_dir / "wave_latent.npz",
        latent=latent,
        mask=data["mask"],
        lane=data["lane"],
        match_id=data["match_id"],
    )
    with open(settings.model_dir / "wave_state_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
