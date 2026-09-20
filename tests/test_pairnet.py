import numpy as np
import torch

from synergy.ml.pairnet import (
    COMBINATIONS,
    InteractionNet,
    centred_fit,
    corpus_fits,
    corpus_pairs,
    ensemble_side,
    explained,
    family_masks,
    gather,
    linear_predict,
    linear_weights,
    pair_index,
    pair_target,
    partner_means,
    player_means,
    reference_players,
    save_ensemble,
    shape,
    shuffle_partners,
    train_interaction,
)
from synergy.ml.serving import _npz, fits_between, network_side


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


def test_family_masks_zero_one_family_at_a_time_in_what_the_network_sees():
    # given
    columns = ["tend_a", "tend_b", "prio_a", "rsp_a"]
    reduced = torch.ones(2, 10, 4)
    index = torch.tensor([[0, 0, 0, 1, 0, 5, 0, 6], [1, 0, 1, 1, 1, 5, 1, 6]])

    # when
    masks = family_masks(columns, "cpu")
    picked = gather(reduced, index, masks["tend"])

    # then
    assert set(masks) == {"tend", "prio", "rsp"}
    assert masks["tend"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert picked[:, :, :2].abs().sum() == 0 and picked[:, :, 2:].sum() == 2 * 4 * 2


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


def test_the_saved_ensemble_serves_the_same_fit_the_networks_compute(tmp_path):
    # given
    torch.manual_seed(6)
    rng = np.random.default_rng(6)
    columns = [f"tend_{k}" for k in range(3)] + [f"prio_{k}" for k in range(3)]
    models = [(InteractionNet(6, hidden=16).eval(), 40.0), (InteractionNet(6, hidden=16).eval(), 60.0)]
    reference = rng.normal(size=(5, 8, 6)).astype(np.float32)
    basis = {"centre": np.zeros(6), "spread": np.ones(6)}

    # when
    spread = save_ensemble(tmp_path / "pairnet.npz", models, basis, columns, reference, np.zeros(6), 0.0, "cpu", lineups=200)
    net = _npz(tmp_path / "pairnet.npz")
    first, second = rng.normal(size=(4, 6)).astype(np.float32), rng.normal(size=(4, 6)).astype(np.float32)
    served = network_side(net, first, second)
    trained = ensemble_side(models, torch.as_tensor(first), torch.as_tensor(second)).numpy()
    grid, grand, by_seed = centred_fit(models, torch.as_tensor(reference[1]), torch.as_tensor(reference[3]))
    fits, error = fits_between(net, reference[1][2], reference[3][5], 1, 3)

    # then
    assert np.allclose(served, trained, atol=1e-5)
    assert np.isclose(fits["all"], float(grid[2, 5]), atol=1e-4)
    assert set(fits) == {"all", "prio", "tend"} and error > 0.0
    assert np.isclose(by_seed.mean(), grand, atol=1e-5) and net["grand_seeds"].shape == (10, 2)
    assert len(spread) == 10 and net["fit_quantiles"].shape == (10, 5, 1001) and net["team_quantiles"].shape == (1001,)
    assert [str(name) for name in net["combos"]][0] == "TOP+JUNGLE"


def test_corpus_pairs_are_centred_against_the_same_reference_players_serving_uses():
    # given
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    matches, positions = 6, ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    basis = {
        "reduced": rng.normal(size=(matches, 10, 4)).astype(np.float32),
        "seat_puuid": np.array([[f"p{seat}" for seat in range(10)]] * matches),
        "seat_position": np.array([positions * 2] * matches),
    }
    blue, red = np.tile(np.arange(5), (matches, 1)), np.tile(np.arange(5, 10), (matches, 1))
    models = [(InteractionNet(4, hidden=8).eval(), 1.0)]

    # when
    table = player_means(basis, np.arange(matches))
    reference = reference_players(table, rng, count=2, least=1)
    corpus = corpus_pairs(basis, np.arange(matches), blue, red, table["at"])
    partner = partner_means(models, table, reference, "cpu")
    grands = np.array([centred_fit(models, torch.as_tensor(reference[a]), torch.as_tensor(reference[b]))[1] for a, b in COMBINATIONS])
    fits = corpus_fits(models, table, corpus, partner, grands, "cpu")
    a, b = table["at"]["p0|TOP"], table["at"]["p1|JUNGLE"]
    own = ensemble_side(models, torch.as_tensor(table["means"][[a]]), torch.as_tensor(table["means"][[b]]))[0]
    with_others = ensemble_side(models, torch.as_tensor(np.repeat(table["means"][[a]], 2, 0)), torch.as_tensor(reference[1])).mean()
    others_with = ensemble_side(models, torch.as_tensor(reference[0]), torch.as_tensor(np.repeat(table["means"][[b]], 2, 0))).mean()

    # then
    assert len(table["at"]) == 10 and table["counts"].tolist() == [matches] * 10
    assert corpus["combo"].shape == (matches * 20,) and corpus["side"].max() == matches * 2 - 1
    assert np.isclose(table["means"][a], basis["reduced"][:, 0].mean(axis=0), atol=1e-6).all()
    assert np.isclose(fits[0], float(own - with_others - others_with) + grands[0], atol=1e-5)


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
    guess, _, _, _ = train_interaction(
        tensor, learn, values[:4000] - linear["learn"], check, values[4000:5000] - linear["check"], held, 0, epochs=15
    )

    # then
    alone = explained(target[5000:], linear["held"].numpy())
    together = explained(target[5000:], (linear["held"] + guess).numpy())
    assert alone > 0.3
    assert together > alone + 0.5 * (1.0 - alone)
