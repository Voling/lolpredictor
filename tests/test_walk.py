import numpy as np
import torch

from synergy.deep.walk import MatchWalk, batches


def _batch(rows: int = 2, span: int = 8):
    rng = np.random.default_rng(0)
    top = rng.integers(0, 19, size=(rows, span, 3)).astype(np.int8)
    probability = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=(rows, span))[..., :3].astype(np.float16)
    return {
        "kind": rng.integers(0, 21, size=(rows, span)).astype(np.int16),
        "actor": rng.integers(0, 11, size=(rows, span)).astype(np.int16),
        "victim": rng.integers(0, 11, size=(rows, span)).astype(np.int16),
        "region": rng.integers(0, 19, size=(rows, span)).astype(np.int16),
        "seat_region_top": top,
        "seat_region_p": probability,
        "wave": rng.dirichlet([1.0, 1.0, 1.0], size=(rows, span)).astype(np.float16),
        "lead": rng.normal(size=(rows, span)).astype(np.float32),
        "clock": np.linspace(0.0, 1.0, span, dtype=np.float32)[None].repeat(rows, axis=0),
        "mask": np.ones((rows, span), bool),
        "seat_champion": rng.integers(0, 5, size=(rows, 10)).astype(np.int16),
        "seat_role": rng.integers(0, 6, size=(rows, 10)).astype(np.int16),
        "seat_side": np.array([[1] * 5 + [0] * 5] * rows, np.int8),
        "win": np.array([1, 0], np.int8),
    }


def test_the_walk_consumes_a_region_posterior_and_a_wave_channel():
    # given
    torch.manual_seed(0)
    data = _batch()
    model = MatchWalk(kinds=21, regions=19, champions=5, span=8, dim=16, heads=2, layers=1)

    # when
    *inputs, target = next(batches(data, np.arange(2), 2, "cpu", shuffle=False))
    logit = model(*inputs)
    seats = model.seat_encodings(*inputs)

    # then
    assert logit.shape == (2,) and torch.isfinite(logit).all()
    assert seats.shape == (2, 10, 16) and torch.isfinite(seats).all()
    assert target.tolist() == [1.0, 0.0]


def test_leftover_posterior_mass_falls_on_the_unknown_region():
    # given
    torch.manual_seed(0)
    model = MatchWalk(kinds=21, regions=19, champions=5, span=1, dim=8, heads=2, layers=1)
    data = _batch(rows=1, span=1)
    data["seat_region_top"][:] = 0
    data["seat_region_p"][:] = 0.0

    # when
    *inputs, _ = next(batches(data, np.arange(1), 1, "cpu", shuffle=False))
    top, probability = inputs[4], inputs[5]
    where = (model.seat_region(top) * probability.unsqueeze(-1)).sum(dim=2)
    where = where + model.seat_region.weight[0] * (1.0 - probability.sum(dim=-1, keepdim=True))

    # then
    assert torch.allclose(where[0, 0], model.seat_region.weight[0])
