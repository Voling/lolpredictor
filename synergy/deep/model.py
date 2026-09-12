import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, strength):
        ctx.strength = strength
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def reverse_gradient(tensor: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReversal.apply(tensor, strength)


class TimelineEncoder(nn.Module):
    def __init__(
        self,
        regions: int,
        event_regions: int,
        numeric: int,
        minutes: int,
        champions: int,
        roles: int,
        dim: int = 128,
        embed_dim: int = 64,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        identity_conditioning: bool = False,
    ):
        super().__init__()
        self.identity_conditioning = identity_conditioning
        region_dim, event_dim = 48, 32
        self.region_embedding = nn.Embedding(regions, region_dim)
        self.event_embedding = nn.Embedding(event_regions, event_dim)
        self.numeric_projection = nn.Linear(numeric, dim - region_dim - event_dim)
        self.position_embedding = nn.Embedding(minutes + 1, dim)
        self.champion_embedding = nn.Embedding(champions, dim)
        self.role_embedding = nn.Embedding(roles, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed_dim))

    def forward(self, regions, event_regions, numeric, mask, champion, role):
        batch = regions.shape[0]
        tokens = torch.cat(
            [
                self.region_embedding(regions),
                self.event_embedding(event_regions),
                self.numeric_projection(numeric),
            ],
            dim=-1,
        )
        cls = self.cls.expand(batch, -1, -1)
        if self.identity_conditioning:
            cls = cls + (self.champion_embedding(champion) + self.role_embedding(role))[:, None]
        sequence = torch.cat([cls, tokens], dim=1)
        steps = torch.arange(sequence.shape[1], device=regions.device)
        sequence = self.input_norm(sequence + self.position_embedding(steps)[None])
        padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=regions.device), ~mask], dim=1
        )
        encoded = self.encoder(self.dropout(sequence), src_key_padding_mask=padding)
        return nn.functional.normalize(self.head(encoded[:, 0]), dim=-1)


class Adversary(nn.Module):
    def __init__(self, embed_dim: int, classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes)
        )

    def forward(self, embedding: torch.Tensor, strength: float) -> torch.Tensor:
        return self.net(reverse_gradient(embedding, strength))


def centroid_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    unique, inverse = torch.unique(labels, return_inverse=True)
    one_hot = torch.zeros(len(labels), len(unique), device=embeddings.device)
    one_hot[torch.arange(len(labels)), inverse] = 1.0
    counts = one_hot.sum(dim=0)
    totals = one_hot.T @ embeddings
    held_out = (totals[inverse] - embeddings) / (counts[inverse] - 1).clamp_min(1.0)[:, None]
    centroids = nn.functional.normalize(totals / counts[:, None].clamp_min(1.0), dim=-1)
    usable = counts[inverse] > 1
    if not usable.any():
        return embeddings.sum() * 0.0
    anchors = nn.functional.normalize(held_out[usable], dim=-1)
    logits = anchors @ centroids.T / temperature
    return nn.functional.cross_entropy(logits, inverse[usable])


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    similarity = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(identity, float("-inf"))
    positives = (labels[:, None] == labels[None, :]) & ~identity
    log_probabilities = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    counts = positives.sum(dim=1)
    usable = counts > 0
    if not usable.any():
        return embeddings.sum() * 0.0
    masked = torch.where(positives, log_probabilities, torch.zeros_like(log_probabilities))
    per_anchor = masked.sum(dim=1)[usable] / counts[usable]
    return -per_anchor.mean()
