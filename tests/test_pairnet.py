import numpy as np
import torch

from synergy.ml.pairnet import (
    COMBINATIONS,
    CurvedNet,
    InteractionNet,
    explained,
    gather,
    linear_predict,
    linear_weights,
    pair_index,
    pair_target,
    shape,
    split_rows,
    train_interaction,
)


def _seats(rng, matches):
    blue = np.tile(np.arange(5), (matches, 1))
    red = np.tile(np.arange(5, 10), (matches, 1))
    reduced = rng.normal(size=(matches, 10, 6)).astype(np.float32)
    return blue, red, reduced


def test_the_interaction_flips_sign_when_the_sides_swap():
    # given
    torch.manual_seed(0)
    model = InteractionNet(6)
    model.eval()
    cells = torch.randn(3, 4, 6)
    swapped = cells[:, [2, 3, 0, 1]]

    # when
    forward, backward = model(cells), model(swapped)

    # then
    assert torch.allclose(forward, -backward, atol=1e-6)


def test_the_curved_net_cannot_express_a_pairing():
    # given
    torch.manual_seed(1)
    model = CurvedNet(6)
    model.eval()
    a, b, c, d = torch.randn(4, 5, 6)

    # when
    with_b, with_c = model.side(a, b), model.side(a, c)
    other_with_b, other_with_c = model.side(d, b), model.side(d, c)

    # then
    assert torch.allclose(with_b - with_c, other_with_b - other_with_c, atol=1e-5)


def test_pair_rows_index_the_two_seats_and_their_mirrors_for_every_combination():
    # given
    rng = np.random.default_rng(1)
    blue, red, reduced = _seats(rng, 3)
    gold = rng.normal(size=(3, 10)) * 100.0

    # when
    index = pair_index(blue, red, np.arange(3))
    target = pair_target(gold, index)

    # then
    assert index.shape == (30, 8)
    assert index[0].tolist() == [0, 0, 0, 1, 0, 5, 0, 6]
    assert np.isclose(target[0], gold[0, 0] + gold[0, 1] - gold[0, 5] - gold[0, 6])
    picked = gather(torch.as_tensor(reduced), torch.as_tensor(index[:2]))
    assert picked.shape == (2, 4, 6) and torch.allclose(picked[0, 3], torch.as_tensor(reduced[0, 6]))


def test_holding_premades_out_keeps_every_premade_row_from_the_fit():
    # given
    rng = np.random.default_rng(8)
    blue, red, _ = _seats(rng, 12)
    everyone = np.arange(12)
    all_raw = pair_index(blue, red, everyone)
    combo_all = np.repeat(np.arange(len(COMBINATIONS)), len(everyone))
    match_all = np.tile(everyone, len(COMBINATIONS))
    ours_all = (match_all % 4 == 0) & (combo_all == 0)
    learn, check, test = np.arange(0, 8), np.arange(8, 10), np.arange(10, 12)

    # when
    plain, plain_combo, plain_ours = split_rows(all_raw, combo_all, match_all, ours_all, learn, check, test, False)
    held, held_combo, held_ours = split_rows(all_raw, combo_all, match_all, ours_all, learn, check, test, True)

    # then
    assert len(plain["learn"]) == 80 and len(plain["held"]) == 20 and plain_ours.sum() == 0
    assert len(held["learn"]) == 78 and len(held["held"]) == 23 and held_ours.sum() == 3
    assert (held_combo["held"][held_ours] == 0).all() and set(held["held"][held_ours, 0]) == {0, 4, 8}


def test_shape_reports_the_spread_and_the_tails_of_a_distribution():
    # given
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(size=10000), [80.0, -80.0]])

    # when
    found = shape(values)

    # then
    assert 0.9 < found["spread"] < 1.6
    assert found["kurtosis"] > 3.5
    assert found["widest"] > 70.0


def test_the_network_recovers_a_planted_interaction_the_curved_net_cannot():
    # given
    rng = np.random.default_rng(4)
    matches = 600
    blue, red, reduced = _seats(rng, matches)
    index = pair_index(blue, red, np.arange(matches))
    cells = reduced[index[:, 0::2], index[:, 1::2]]
    linear_part = (cells[:, 0] + cells[:, 1] - cells[:, 2] - cells[:, 3]) @ np.arange(6)
    interaction = (cells[:, 0, :3] * cells[:, 1, :3]).sum(axis=1) - (cells[:, 2, :3] * cells[:, 3, :3]).sum(axis=1)
    target = linear_part + 3.0 * interaction + rng.normal(scale=0.5, size=len(index))
    tensor = torch.as_tensor(reduced)
    learn, check, held = (torch.as_tensor(index[a:b]) for a, b in ((0, 4000), (4000, 5000), (5000, 6000)))
    values = torch.as_tensor(target, dtype=torch.float32)

    # when
    weights, middle = linear_weights(tensor, learn, values[:4000], check, values[4000:5000])
    linear = {name: linear_predict(tensor, rows, weights, middle) for name, rows in (("learn", learn), ("check", check), ("held", held))}
    residuals = (values[:4000] - linear["learn"], values[4000:5000] - linear["check"])
    network, _ = train_interaction(tensor, learn, residuals[0], check, residuals[1], held, 0, epochs=15)
    curved, _ = train_interaction(tensor, learn, residuals[0], check, residuals[1], held, 0, epochs=15, build=CurvedNet)

    # then
    alone = explained(target[5000:], linear["held"].numpy())
    with_network = explained(target[5000:], (linear["held"] + network).numpy())
    with_curved = explained(target[5000:], (linear["held"] + curved).numpy())
    assert alone > 0.3
    assert with_network > alone + 0.5 * (1.0 - alone)
    assert with_network > with_curved + 0.3 * (1.0 - alone)
