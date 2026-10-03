import numpy as np
import pandas as pd

from synergy.features.cells import SeatIndex, cells_frame, evidence_shares, fit_cells
from synergy.features.fits import apply_cells, apply_tendencies, fit_kind, load_cells, load_tendencies, save_cells, save_tendencies
from synergy.features.propensity import COVARIATES, KINDS
from synergy.features.tendency import CONTEXT, TENDENCY_COLUMNS, leave_one_out_ratio


def _seats(players=12, matches=6):
    rows = []
    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    for match in range(matches):
        for seat in range(10):
            rows.append({"match_id": f"m{match}", "puuid": f"p{(seat + match) % players}", "position": positions[seat % 5]})
    return pd.DataFrame(rows)


def test_saved_cells_reproduce_the_corpus_values_and_evidence_for_the_same_seats(tmp_path):
    # given
    rng = np.random.default_rng(1)
    index = SeatIndex(_seats())
    situations, states, outcomes = ["near", "far"], ["early_behind", "late_ahead"], ["a", "b", "c"]
    split = {"prefix": "objg", "situations": ["far"], "groups": {"behind": ["early_behind"], "ahead": ["late_ahead"]}}
    counts = rng.poisson(3.0, size=(len(index), len(situations), len(states), len(outcomes))).astype(float)
    fit = fit_cells(counts, index, situations, outcomes, "obj", scale=1.5, unit=2.0, kappa=[4.0, 6.0], states=states, split=split)
    expected = cells_frame([(fit, counts)], index, fit.columns)
    save_cells(tmp_path / "cells.npz", fit, index.positions)

    # when
    saved = load_cells(tmp_path / "cells.npz")
    frame, evidence = apply_cells(saved, counts, index)

    # then
    assert saved["states"] == states and saved["worlds"].shape == (2, 5, 2, 3) and saved["split"] == split
    assert list(frame.columns[-6:]) == [f"objg_{gold}_far_{outcome}" for gold in ("behind", "ahead") for outcome in outcomes]
    pd.testing.assert_frame_equal(frame, expected)
    pd.testing.assert_frame_equal(evidence, evidence_shares(fit, index))


def test_a_new_player_is_scored_against_the_saved_world_with_their_own_leave_one_out_totals(tmp_path):
    # given
    rng = np.random.default_rng(2)
    corpus = SeatIndex(_seats())
    counts = rng.poisson(3.0, size=(len(corpus), 1, 3)).astype(float)
    fit = fit_cells(counts, corpus, ["all"], ["a", "b", "c"], "rsp", kappa=[5.0])
    save_cells(tmp_path / "cells.npz", fit, corpus.positions)
    stranger = SeatIndex(pd.DataFrame({"match_id": ["x1", "x2", "x3"], "puuid": ["new"] * 3, "position": ["TOP", "TOP", "SQUID"]}))
    own = np.array([[[2.0, 0.0, 0.0]], [[0.0, 4.0, 0.0]], [[1.0, 1.0, 1.0]]])

    # when
    frame, evidence = apply_cells(load_cells(tmp_path / "cells.npz"), own, stranger)

    # then
    world = fit.worlds[0, corpus.positions.index("TOP"), 0]
    others = np.array([0.0, 4.0, 0.0])
    assert np.allclose(frame.iloc[0, 2:].to_numpy(dtype=float), (others - 4.0 * world) / (4.0 + 5.0))
    assert frame.iloc[2, 2:].isna().all()
    assert set(evidence["position"]) == {"TOP", "SQUID"} and evidence.loc[evidence.position == "TOP", "share"].iloc[0] > 0


def test_a_fit_saved_before_states_loads_as_one_state_of_shares_and_scores_seats_as_before(tmp_path):
    # given
    rng = np.random.default_rng(4)
    index = SeatIndex(_seats())
    counts = rng.poisson(3.0, size=(len(index), 2, 3)).astype(float)
    fit = fit_cells(counts, index, ["near", "far"], ["a", "b", "c"], "rsp", kappa=[4.0, 6.0])
    np.savez(
        tmp_path / "old.npz",
        prefix=np.array("rsp"),
        situations=np.array(fit.situations),
        outcomes=np.array(fit.outcomes),
        worlds=fit.worlds[:, :, 0],
        kappa=fit.kappa,
        unit=np.array(1.0),
        positions=np.array(index.positions),
        typical=fit.totals.sum(axis=(2, 3)).mean(axis=0),
    )
    save_cells(tmp_path / "new.npz", fit, index.positions)

    # when
    old, new = load_cells(tmp_path / "old.npz"), load_cells(tmp_path / "new.npz")
    shares, _ = apply_cells(old, counts, index)
    gaps, _ = apply_cells(new, counts, index)

    # then
    assert old["states"] == ["all"] and old["worlds"].shape == (2, 5, 1, 3) and old["gap"] is False and new["gap"] is True
    assert old["split"] is None and new["split"] is None
    others = fit.totals[index.who_code][:, :, 0, :] - counts
    world = fit.worlds[:, index.seat_position, 0].transpose(1, 0, 2)
    kappa = fit.kappa[None, :, None]
    expected = (world * kappa + others) / (others.sum(axis=2, keepdims=True) + kappa)
    assert np.allclose(shares.iloc[:, 2:].to_numpy(dtype=float), expected.reshape(len(index), -1))
    assert np.allclose(gaps.iloc[:, 2:].to_numpy(dtype=float), (expected - world).reshape(len(index), -1))


def _opportunities(rng, seats, kind):
    rows = []
    for _, seat in seats.iterrows():
        for minute in range(3, 13):
            rows.append(
                {
                    "match_id": seat.match_id,
                    "puuid": seat.puuid,
                    "kind": kind,
                    "minute": minute,
                    "gold_diff": rng.normal(),
                    "is_jungler": float(seat.position == "JUNGLE"),
                    "in_own_half": float(rng.random() < 0.5),
                    "distance_to_enemy_jungle": rng.random(),
                    "unspent_gold": rng.random(),
                    "role": seat.position,
                    "outcome": float(rng.random() < 0.3 + 0.1 * (seat.position == "JUNGLE")),
                }
            )
    return pd.DataFrame(rows)


def test_saved_tendencies_reproduce_the_build_for_the_same_rows_and_fill_missing_context_like_the_corpus(tmp_path):
    # given
    rng = np.random.default_rng(3)
    seats = _seats()
    kind = KINDS[0]
    opportunities = _opportunities(rng, seats, kind)
    context = pd.DataFrame(
        {"match_id": seats.match_id, "puuid": seats.puuid, "minute": 5, **{column: rng.random(len(seats)) for column in CONTEXT}}
    )
    fill = {column: float(context[column].mean()) for column in CONTEXT}
    from synergy.features.fits import _with_context

    subset = _with_context(opportunities, context, fill).merge(seats, on=["match_id", "puuid"], how="inner")
    everyone = seats[["puuid", "position"]].drop_duplicates().set_index(["puuid", "position"]).index
    fit, frame = fit_kind(subset, everyone, 1.0, fill)
    expected = leave_one_out_ratio(frame, seats, fit["prior"], f"tend_{kind}")
    save_tendencies(tmp_path / "fit.pkl", {kind: fit})

    # when
    out, evidence = apply_tendencies(load_tendencies(tmp_path / "fit.pkl"), opportunities, context, seats)

    # then
    merged = out.merge(expected, on=["match_id", "puuid"], suffixes=("", "_build"))
    for column in [c for c in expected.columns if c.startswith("tend_")]:
        assert np.allclose(merged[column], merged[f"{column}_build"])
    assert set(TENDENCY_COLUMNS) <= set(out.columns) and len(evidence) == len(everyone)
    assert COVARIATES[0] in fit["columns"] and any(column.startswith("role_") for column in fit["columns"])


def test_tendency_totals_sum_what_was_seen_and_expected_per_player_and_situation():
    # given
    from synergy.features.fits import _with_context, tendency_totals
    from synergy.features.tendency import CELLS

    rng = np.random.default_rng(5)
    seats = _seats()
    kind = KINDS[0]
    opportunities = _opportunities(rng, seats, kind)
    context = pd.DataFrame({"match_id": seats.match_id, "puuid": seats.puuid, "minute": 5, **{column: rng.random(len(seats)) for column in CONTEXT}})
    fill = {column: float(context[column].mean()) for column in CONTEXT}
    subset = _with_context(opportunities, context, fill).merge(seats, on=["match_id", "puuid"], how="inner")
    everyone = seats[["puuid", "position"]].drop_duplicates().set_index(["puuid", "position"]).index
    fit, frame = fit_kind(subset, everyone, 1.0, fill)

    # when
    totals = tendency_totals({kind: fit}, lambda wanted: opportunities[opportunities["kind"] == wanted], context, seats)

    # then
    sums = frame.groupby(["puuid", "position", "cell"])[["observed", "expected"]].sum()
    who, cell = (frame.puuid.iloc[0], frame.position.iloc[0]), frame.cell.iloc[0]
    row = totals.set_index(["puuid", "position"]).loc[who]
    assert np.isclose(row[f"tend_{kind}_{cell}_obs"], sums.loc[(*who, cell), "observed"]) and np.isclose(row[f"tend_{kind}_{cell}_exp"], sums.loc[(*who, cell), "expected"])
    assert len(totals) == len(everyone) and all(f"tend_{kind}_{name}_{part}" in totals.columns for name in CELLS for part in ("obs", "exp"))
