import numpy as np
import pytest

torch = pytest.importorskip("torch")

from synergy.deep.model import supervised_contrastive_loss
from synergy.features.regions import REGIONS, region_of
from synergy.deep.train import active_subset, augment, make_splits, retrieval_metrics


def test_regions_are_mirrored_for_the_red_team():
    blue_jungle = (5200.0, 8200.0)
    assert REGIONS[region_of(*blue_jungle, 100)] == "JUNGLE_OWN_TOPSIDE"
    assert REGIONS[region_of(*blue_jungle, 200)] == "JUNGLE_ENEMY_TOPSIDE"


def test_lane_depth_is_relative_to_the_team():
    near_blue_base = (3000.0, 4600.0)
    assert REGIONS[region_of(*near_blue_base, 100)] == "LANE_MID_OWN"
    assert REGIONS[region_of(*near_blue_base, 200)] == "LANE_MID_ENEMY"


def test_pits_and_bases_are_distinct():
    assert REGIONS[region_of(4400.0, 10200.0, 100)] == "RIVER_BARON"
    assert REGIONS[region_of(9800.0, 4400.0, 100)] == "RIVER_DRAGON"
    assert REGIONS[region_of(1500.0, 1500.0, 100)] == "BASE_OWN"
    assert REGIONS[region_of(1500.0, 1500.0, 200)] == "BASE_ENEMY"


def test_missing_coordinates_fall_back_to_unknown():
    assert REGIONS[region_of(0.0, 0.0, 100)] == "UNKNOWN"


def test_contrastive_loss_rewards_matching_pairs():
    labels = torch.tensor([0, 0, 1, 1])
    close = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]]), dim=-1
    )
    scattered = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.1], [0.1, 0.99]]), dim=-1
    )
    assert supervised_contrastive_loss(close, labels) < supervised_contrastive_loss(scattered, labels)


def test_contrastive_loss_is_finite_without_positives():
    labels = torch.tensor([0, 1, 2])
    embeddings = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    assert torch.isfinite(supervised_contrastive_loss(embeddings, labels))


def test_splits_hold_out_one_game_for_players_with_three_or_more():
    puuids = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"])
    train, query = make_splits(puuids, seed=1)
    assert len(query) == 2
    assert sorted(puuids[query]) == ["a", "b"]
    assert len(train) == len(puuids) - len(query)
    assert set(train).isdisjoint(set(query))


def test_active_subset_keeps_only_well_observed_players():
    puuids = np.array(["a"] * 5 + ["b"] * 2)
    train, _ = make_splits(puuids, seed=1)
    active = active_subset(puuids, train, minimum=3)
    assert set(puuids[active]) == {"a"}


def test_retrieval_is_perfect_when_queries_match_their_gallery():
    puuids = np.array(["a", "a", "a", "b", "b", "b"])
    embeddings = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    train, query = make_splits(puuids, seed=3)
    recall1, recall5, mrr, chance = retrieval_metrics(embeddings, puuids, train, query)
    assert recall1 == 1.0
    assert recall5 == 1.0
    assert mrr == 1.0
    assert chance == pytest.approx(0.5)


def test_augment_never_widens_the_mask():
    mask = torch.zeros(8, 20, dtype=torch.bool)
    mask[:, :16] = True
    view = augment(mask, crop_minimum=0.6, step_dropout=0.1)
    assert bool((view & ~mask).sum() == 0)
    assert int(view.sum(dim=1).min()) >= 4


def test_augment_is_a_noop_when_disabled():
    mask = torch.zeros(4, 12, dtype=torch.bool)
    mask[:, :10] = True
    assert torch.equal(augment(mask, crop_minimum=1.0, step_dropout=0.0), mask)
