import numpy as np
import pandas as pd

from synergy.features.positions import POSITIONS
from synergy.ml.mirrored import (
    Board,
    directions,
    explained,
    pair_design,
    pair_swings,
    ridge,
    seat_games,
    seats_by_position,
)
from synergy.ingest.premades import sessions


def test_each_seat_is_matched_with_the_same_position_on_the_other_side():
    # given
    positions = np.array([["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2])
    side = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    # when
    blue, red = seats_by_position(positions, side)

    # then
    assert blue.tolist() == [[0, 1, 2, 3, 4]]
    assert red.tolist() == [[5, 6, 7, 8, 9]]


def test_a_team_missing_a_position_is_marked_rather_than_matched():
    # given
    positions = np.array([["TOP", "TOP", "MIDDLE", "BOTTOM", "UTILITY", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]])
    side = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    # when
    blue, red = seats_by_position(positions, side)

    # then
    assert blue[0, POSITIONS.index("JUNGLE")] == -1
    assert (red[0] >= 0).all()


def _board(rng, matches=4, swings=None, premade=None):
    reduced = rng.normal(size=(matches, 10, 6))
    gold = rng.normal(size=(matches, 10)) * 1000.0
    positions = np.array([["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2] * matches)
    side = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * matches)
    blue, red = seats_by_position(positions, side)
    names = np.array([[f"p{seat}" for seat in range(10)]] * matches)
    return Board(
        reduced=reduced,
        gold=gold,
        blue=blue,
        red=red,
        games=seat_games(names, positions),
        match_id=np.array([f"m{row}" for row in range(matches)]),
        seat_puuid=names,
        premade=premade or set(),
        swings=swings,
    )


def test_pair_rows_carry_the_two_seats_against_their_mirrors():
    # given
    rng = np.random.default_rng(0)
    board = _board(rng, premade={("p0", "p1")})
    projection = directions(board.reduced, np.arange(4), 3)
    upper = np.triu_indices(3)

    # when
    rows = pair_design(board, projection, np.arange(4), upper)

    # then
    assert rows.additive.shape == (40, 6) and rows.products.shape == (40, 6) and rows.target.shape == (40,)
    assert np.allclose(rows.target[0], board.gold[0, 0] + board.gold[0, 1] - board.gold[0, 5] - board.gold[0, 6])
    assert np.allclose(rows.additive[0], board.reduced[0, 0] + board.reduced[0, 1] - board.reduced[0, 5] - board.reduced[0, 6])
    assert rows.depth.tolist() == [4] * 40
    assert rows.premade[0] and not rows.premade[4]


def test_the_event_target_sums_what_the_pair_swung_less_what_their_mirrors_swung():
    # given
    rng = np.random.default_rng(1)
    swings = {("m0", "p0", "p1"): 450.0, ("m0", "p5", "p6"): -175.0}
    board = _board(rng, swings=swings)
    projection = directions(board.reduced, np.arange(4), 3)

    # when
    rows = pair_design(board, projection, np.arange(4), np.triu_indices(3))

    # then
    assert rows.target[0] == 625.0
    assert rows.target[1] == 0.0


def test_pair_swings_credit_both_players_present_at_an_event_with_its_signed_value():
    # given
    frame = pd.DataFrame(
        {
            "match_id": ["m"] * 5,
            "trigger_id": [0, 0, 0, 1, 1],
            "trigger": ["kill", "kill", "kill", "objective", "objective"],
            "detail": ["", "", "", "DRAGON", "DRAGON"],
            "puuid": ["a", "b", "x", "a", "b"],
            "team_id": [100, 100, 200, 100, 100],
            "ours": [1, 1, 0, 0, 0],
        }
    )

    # when
    swings = pair_swings(frame, {"DRAGON": 600.0})

    # then
    assert swings == {("m", "a", "b"): 450.0 - 600.0}


def test_sessions_flag_pairs_who_queue_together_in_close_succession():
    # given
    when = pd.to_datetime(["2026-03-01 20:00", "2026-03-01 20:40", "2026-03-01 21:20", "2026-06-01 10:00"], utc=True)
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"],
            "puuid": ["a", "b", "a", "b", "a", "b", "a", "c"],
            "team_id": [100] * 8,
            "game_creation": [when[0], when[0], when[1], when[1], when[2], when[2], when[3], when[3]],
        }
    )

    # when
    found = sessions(frame)

    # then
    assert found == {("a", "b")}


def test_the_ridge_recovers_a_planted_interaction_that_the_cells_alone_cannot():
    # given
    rng = np.random.default_rng(3)
    rows, width = 4000, 8
    products = rng.normal(size=(rows, width))
    cells = rng.normal(size=(rows, 5))
    truth = rng.normal(size=width)
    values = cells @ rng.normal(size=5) + products @ truth * 2.0 + rng.normal(size=rows) * 0.5
    fit, test = np.arange(3000), np.arange(3000, rows)

    # when
    alone = ridge(cells[fit].T @ cells[fit], cells[fit].T @ values[fit], np.repeat(1.0, 5))
    design = np.hstack([cells, products])
    both = ridge(design[fit].T @ design[fit], design[fit].T @ values[fit], np.repeat(1.0, width + 5))

    # then
    without = explained(values[test], cells[test] @ alone)
    with_products = explained(values[test], design[test] @ both)
    assert with_products > without + 0.2
    assert np.corrcoef(both[5:], truth)[0, 1] > 0.9
