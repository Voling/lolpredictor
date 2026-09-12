import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..config import Settings, get_settings
from ..features.player import STYLE_COLUMNS, normaliser, participation_styles
from .model import Adversary, TimelineEncoder, centroid_loss, supervised_contrastive_loss
from .sequences import MAX_MINUTES, NO_EVENT_REGION, load_sequences
from ..features.regions import REGIONS

logger = logging.getLogger(__name__)


@dataclass
class DeepReport:
    sequences: int = 0
    train_games: int = 0
    query_games: int = 0
    identities: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    champion_accuracy: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mean_reciprocal_rank: float = 0.0
    baseline_recall_at_1: float = 0.0
    baseline_recall_at_5: float = 0.0
    baseline_mean_reciprocal_rank: float = 0.0
    chance_recall_at_1: float = 0.0
    active_recall_at_1: float = 0.0
    active_baseline_recall_at_1: float = 0.0
    active_gallery: int = 0
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sequences": self.sequences,
            "train_games": self.train_games,
            "query_games": self.query_games,
            "identities": self.identities,
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 4),
            "champion_probe_accuracy": round(self.champion_accuracy, 4),
            "retrieval": {
                "recall_at_1": round(self.recall_at_1, 4),
                "recall_at_5": round(self.recall_at_5, 4),
                "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            },
            "handwritten_axes_baseline": {
                "recall_at_1": round(self.baseline_recall_at_1, 4),
                "recall_at_5": round(self.baseline_recall_at_5, 4),
                "mean_reciprocal_rank": round(self.baseline_mean_reciprocal_rank, 4),
            },
            "chance_recall_at_1": round(self.chance_recall_at_1, 6),
            "active_gallery_players": self.active_gallery,
            "active_gallery_recall_at_1": round(self.active_recall_at_1, 4),
            "active_gallery_baseline_recall_at_1": round(self.active_baseline_recall_at_1, 4),
            "history": self.history,
        }


def make_splits(puuids: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_player: dict[str, list[int]] = {}
    for index, puuid in enumerate(puuids):
        by_player.setdefault(puuid, []).append(index)
    train, query = [], []
    for indices in by_player.values():
        if len(indices) >= 3:
            held = int(rng.choice(indices))
            query.append(held)
            train.extend(i for i in indices if i != held)
        else:
            train.extend(indices)
    return np.array(sorted(train)), np.array(sorted(query))


def retrieval_metrics(
    embeddings: np.ndarray, puuids: np.ndarray, train: np.ndarray, query: np.ndarray
) -> tuple[float, float, float, float]:
    gallery: dict[str, list[int]] = {}
    for index in train:
        gallery.setdefault(puuids[index], []).append(index)
    names = sorted(gallery)
    if not names or len(query) == 0:
        return 0.0, 0.0, 0.0, 0.0
    centroids = np.stack([embeddings[gallery[name]].mean(axis=0) for name in names])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    lookup = {name: position for position, name in enumerate(names)}
    hits1 = hits5 = 0
    reciprocal = 0.0
    counted = 0
    for index in query:
        target = lookup.get(puuids[index])
        if target is None:
            continue
        counted += 1
        scores = centroids @ embeddings[index]
        order = np.argsort(-scores)
        rank = int(np.where(order == target)[0][0]) + 1
        hits1 += rank == 1
        hits5 += rank <= 5
        reciprocal += 1.0 / rank
    if counted == 0:
        return 0.0, 0.0, 0.0, 0.0
    return hits1 / counted, hits5 / counted, reciprocal / counted, 1.0 / len(names)


def active_subset(puuids: np.ndarray, train: np.ndarray, minimum: int = 3) -> np.ndarray:
    counts = pd.Series(puuids[train]).value_counts()
    keep = set(counts[counts >= minimum].index)
    return np.array([index for index in train if puuids[index] in keep])


def _handwritten_embeddings(data: dict, settings: Settings) -> np.ndarray | None:
    path = settings.processed_dir / "participations.parquet"
    if not path.exists():
        return None
    participations = pd.read_parquet(path)
    styles = participation_styles(participations, normaliser(participations))
    styles = styles.set_index(["match_id", "puuid"])[STYLE_COLUMNS]
    keys = pd.MultiIndex.from_arrays([data["match_id"], data["puuid"]])
    aligned = styles.reindex(keys).to_numpy(dtype=np.float32)
    aligned = np.nan_to_num(aligned)
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    return aligned / (norms + 1e-9)


def _batches(puuids: np.ndarray, train: np.ndarray, players: int, per_player: int, rng):
    by_player: dict[str, list[int]] = {}
    for index in train:
        by_player.setdefault(puuids[index], []).append(index)
    usable = [name for name, items in by_player.items() if len(items) >= 2]
    rng.shuffle(usable)
    for start in range(0, len(usable) - players + 1, players):
        chosen = usable[start : start + players]
        batch = []
        for name in chosen:
            items = by_player[name]
            batch.extend(rng.choice(items, size=per_player, replace=len(items) < per_player))
        yield np.array(batch)


def train_encoder(
    settings: Settings | None = None,
    epochs: int = 80,
    dim: int = 128,
    embed_dim: int = 64,
    layers: int = 3,
    players_per_batch: int = 48,
    games_per_player: int = 4,
    centroid_weight: float = 1.0,
    learning_rate: float = 3e-4,
    adversary_strength: float = 1.0,
    identity_conditioning: bool = False,
    crop_minimum: float = 0.6,
    step_dropout: float = 0.1,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    settings = settings or get_settings()
    data = load_sequences(settings)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    puuids = data["puuid"]
    train, query = make_splits(puuids, seed)
    report = DeepReport(
        sequences=len(puuids),
        train_games=len(train),
        query_games=len(query),
        identities=int(sum(1 for _, n in pd.Series(puuids[train]).value_counts().items() if n >= 2)),
        epochs=epochs,
    )

    numeric = torch.from_numpy(data["numeric"]).float()
    mask = torch.from_numpy(data["mask"])
    flat = numeric[torch.from_numpy(train)][mask[torch.from_numpy(train)]]
    mean, std = flat.mean(dim=0), flat.std(dim=0).clamp_min(1e-3)
    numeric = ((numeric - mean) / std) * mask[..., None]

    regions = torch.from_numpy(data["regions"].astype(np.int64)).to(device)
    event_regions = torch.from_numpy(data["event_regions"].astype(np.int64)).to(device)
    numeric = numeric.to(device)
    mask = mask.to(device)
    champions = torch.from_numpy(data["champion"].astype(np.int64)).to(device)
    roles = torch.from_numpy(data["position"].astype(np.int64)).to(device)

    encoder = TimelineEncoder(
        regions=len(REGIONS),
        event_regions=NO_EVENT_REGION + 1,
        numeric=numeric.shape[-1],
        minutes=MAX_MINUTES,
        champions=int(champions.max().item()) + 1,
        roles=int(roles.max().item()) + 1,
        dim=dim,
        embed_dim=embed_dim,
        layers=layers,
        identity_conditioning=identity_conditioning,
    ).to(device)
    adversary = Adversary(embed_dim, int(champions.max().item()) + 1).to(device)
    optimiser = torch.optim.AdamW(
        [*encoder.parameters(), *adversary.parameters()], lr=learning_rate, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=max(epochs * 40, 100), pct_start=0.2
    )
    champion_loss = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        encoder.train()
        adversary.train()
        losses, accuracies = [], []
        strength = adversary_strength * min(1.0, epoch / max(epochs * 0.3, 1))
        for indices in _batches(puuids, train, players_per_batch, games_per_player, rng):
            batch = torch.from_numpy(indices).to(device)
            labels = torch.from_numpy(
                pd.factorize(puuids[indices])[0].astype(np.int64)
            ).to(device)
            view = augment(mask[batch], crop_minimum, step_dropout)
            embeddings = encoder(
                regions[batch],
                event_regions[batch],
                numeric[batch],
                view,
                champions[batch],
                roles[batch],
            )
            contrastive = supervised_contrastive_loss(embeddings, labels)
            if centroid_weight > 0:
                contrastive = contrastive + centroid_weight * centroid_loss(embeddings, labels)
            logits = adversary(embeddings, strength)
            adversarial = champion_loss(logits, champions[batch])
            loss = contrastive + adversarial
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()
            if step < schedule.total_steps - 1:
                schedule.step()
            step += 1
            losses.append(float(contrastive.item()))
            accuracies.append(float((logits.argmax(dim=1) == champions[batch]).float().mean().item()))
        if losses:
            report.final_loss = float(np.mean(losses))
            report.champion_accuracy = float(np.mean(accuracies))
        if epoch % 10 == 0 or epoch == epochs - 1:
            embeddings = infer(encoder, regions, event_regions, numeric, mask, champions, roles, device)
            r1, r5, mrr, chance = retrieval_metrics(embeddings, puuids, train, query)
            report.history.append(
                {
                    "epoch": epoch,
                    "loss": round(report.final_loss, 4),
                    "champion_accuracy": round(report.champion_accuracy, 4),
                    "recall_at_1": round(r1, 4),
                }
            )
            logger.info(
                "epoch %s loss %.4f champion_acc %.3f recall@1 %.3f", epoch, report.final_loss,
                report.champion_accuracy, r1,
            )

    embeddings = infer(encoder, regions, event_regions, numeric, mask, champions, roles, device)
    r1, r5, mrr, chance = retrieval_metrics(embeddings, puuids, train, query)
    report.recall_at_1, report.recall_at_5 = r1, r5
    report.mean_reciprocal_rank, report.chance_recall_at_1 = mrr, chance

    active = active_subset(puuids, train)
    report.active_gallery = int(len(set(puuids[active])))
    report.active_recall_at_1 = retrieval_metrics(embeddings, puuids, active, query)[0]

    handwritten = _handwritten_embeddings(data, settings)
    if handwritten is not None:
        b1, b5, bmrr, _ = retrieval_metrics(handwritten, puuids, train, query)
        report.baseline_recall_at_1, report.baseline_recall_at_5 = b1, b5
        report.baseline_mean_reciprocal_rank = bmrr
        report.active_baseline_recall_at_1 = retrieval_metrics(handwritten, puuids, active, query)[0]

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "config": {
                "dim": dim,
                "embed_dim": embed_dim,
                "layers": layers,
                "numeric": int(numeric.shape[-1]),
                "minutes": MAX_MINUTES,
            },
            "numeric_mean": mean.numpy(),
            "numeric_std": std.numpy(),
        },
        settings.model_dir / "timeline_encoder.pt",
    )
    frame = pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])
    frame.insert(0, "puuid", puuids)
    frame.insert(1, "match_id", data["match_id"])
    frame.to_parquet(settings.processed_dir / "game_embeddings.parquet", index=False)
    per_player = frame.drop(columns=["match_id"]).groupby("puuid").mean()
    per_player.reset_index().to_parquet(settings.processed_dir / "player_embeddings.parquet", index=False)

    payload = report.as_dict()
    payload["device"] = device
    with open(settings.model_dir / "deep_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def augment(mask: torch.Tensor, crop_minimum: float, step_dropout: float) -> torch.Tensor:
    if crop_minimum >= 1.0 and step_dropout <= 0.0:
        return mask
    lengths = mask.sum(dim=1)
    steps = torch.arange(mask.shape[1], device=mask.device)[None]
    keep = torch.rand(len(mask), device=mask.device) * (1.0 - crop_minimum) + crop_minimum
    limit = (lengths * keep).long().clamp_min(6)[:, None]
    start = (torch.rand(len(mask), device=mask.device) * (lengths - limit[:, 0]).clamp_min(0)).long()[:, None]
    window = (steps >= start) & (steps < start + limit)
    view = mask & window
    if step_dropout > 0.0:
        drops = torch.rand(mask.shape, device=mask.device) < step_dropout
        view = view & ~drops
    return torch.where(view.sum(dim=1, keepdim=True) >= 4, view, mask)


@torch.no_grad()
def infer(encoder, regions, event_regions, numeric, mask, champions, roles, device, chunk: int = 1024):
    encoder.eval()
    outputs = []
    for start in range(0, len(regions), chunk):
        stop = start + chunk
        outputs.append(
            encoder(
                regions[start:stop],
                event_regions[start:stop],
                numeric[start:stop],
                mask[start:stop],
                champions[start:stop],
                roles[start:stop],
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs)
