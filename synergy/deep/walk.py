import numpy as np
import torch
from torch import nn

from .stream import MAX_EVENTS, SEATS


class MatchWalk(nn.Module):
    def __init__(
        self,
        kinds: int,
        regions: int,
        champions: int,
        roles: int = 6,
        span: int = MAX_EVENTS,
        dim: int = 128,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kind = nn.Embedding(kinds, dim)
        self.region = nn.Embedding(regions, dim)
        self.seat = nn.Embedding(SEATS + 1, dim, padding_idx=0)
        self.victim_seat = nn.Embedding(SEATS + 1, dim, padding_idx=0)
        self.champion = nn.Embedding(champions, dim)
        self.role = nn.Embedding(roles, dim)
        self.side = nn.Embedding(2, dim)
        self.scalars = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))
        self.seat_region = nn.Embedding(regions, dim)
        self.order = nn.Embedding(span, dim)
        self.blend = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.settle = nn.LayerNorm(dim)
        self.attend = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def seats(self, champion, role, side):
        return self.champion(champion) + self.role(role) + self.side(side)

    def encode(
        self, kind, actor, victim, region, seat_region, lead, clock, mask, champion, role, side
    ):
        roster = self.seats(champion, role, side)
        padded = torch.cat([torch.zeros_like(roster[:, :1]), roster], dim=1)
        who = torch.gather(padded, 1, actor.unsqueeze(-1).expand(-1, -1, padded.size(-1)))
        hurt = torch.gather(padded, 1, victim.unsqueeze(-1).expand(-1, -1, padded.size(-1)))
        token = (
            self.kind(kind)
            + self.region(region)
            + self.seat(actor)
            + self.victim_seat(victim)
            + who
            + hurt
            + self.seat_region(seat_region)
            + self.scalars(torch.stack([clock, lead], dim=-1))
            + self.order(torch.arange(kind.size(1), device=kind.device).unsqueeze(0))
        )
        alive = mask.any(dim=1, keepdim=True)
        safe = mask | ~alive
        encoded = self.settle(self.encoder(self.blend(token), src_key_padding_mask=~safe))
        return encoded, safe

    def forward(self, *inputs):
        encoded, safe = self.encode(*inputs)
        weight = self.attend(encoded).masked_fill(~safe.unsqueeze(-1), float("-inf"))
        weight = torch.softmax(weight, dim=1)
        pooled = (encoded * weight).sum(dim=1)
        average = (encoded * safe.unsqueeze(-1)).sum(dim=1) / safe.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([pooled, average], dim=-1)).squeeze(-1)

    def seat_encodings(self, *inputs):
        encoded, safe = self.encode(*inputs)
        actor = inputs[1]
        score = self.attend(encoded).squeeze(-1)
        owned = actor.unsqueeze(1) == torch.arange(
            1, SEATS + 1, device=actor.device
        ).view(1, SEATS, 1)
        owned = owned & safe.unsqueeze(1)
        weight = score.unsqueeze(1).masked_fill(~owned, float("-inf")).softmax(dim=-1)
        weight = torch.where(owned.any(dim=-1, keepdim=True), weight, torch.zeros_like(weight))
        return torch.einsum("bst,btd->bsd", weight, encoded)


def batches(
    data: dict, index: np.ndarray, size: int, device: str, shuffle: bool = True, target: str = "win"
):
    order = np.random.permutation(index) if shuffle else index
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        yield (
            torch.as_tensor(data["kind"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["actor"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["victim"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["region"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["seat_region"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["lead"][rows].astype(np.float32), device=device),
            torch.as_tensor(data["clock"][rows].astype(np.float32), device=device),
            torch.as_tensor(data["mask"][rows], device=device),
            torch.as_tensor(data["seat_champion"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["seat_role"][rows].astype(np.int64), device=device),
            torch.as_tensor(data["seat_side"][rows].astype(np.int64), device=device),
            torch.as_tensor(data[target][rows].astype(np.float32), device=device),
        )
