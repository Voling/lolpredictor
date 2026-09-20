import numpy as np
import torch

from synergy.ml.pairnet import (
    InteractionNet,
    explained,
    gather,
    linear_predict,
    linear_weights,
    pair_index,
    pair_target,
    shuffle_partners,
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


def test_shuffling_partners_keeps_each_player_and_breaks_the_pairing():
    # given
    rng = np.random.default_rng(2)
    blue, red, _ = _seats(rng, 50)
    index = pair_index(blue, red, np.arange(50))

    # when
    mixed = shuffle_partners(index, np.random.default_rng(3))

    # then
    assert (mixed[:, [0, 1, 4, 5]] == index[:, [0, 1, 4, 5]]).all()
    assert sorted(map(tuple, mixed[:50, [2, 3]].tolist())) == sorted(map(tuple, index[:50, [2, 3]].tolist()))
    assert (mixed[:, 2] != mixed[:, 0]).any()


def test_the_network_recovers_a_planted_interaction_left_over_by_the_ridge():
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
    guess, _ = train_interaction(
        tensor, learn, values[:4000] - linear["learn"], check, values[4000:5000] - linear["check"], held, 0, epochs=15
    )

    # then
    alone = explained(target[5000:], linear["held"].numpy())
    together = explained(target[5000:], (linear["held"] + guess).numpy())
    assert alone > 0.3
    assert together > alone + 0.5 * (1.0 - alone)
