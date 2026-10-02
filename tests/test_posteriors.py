import numpy as np
import pandas as pd

from synergy.ml import serving
from synergy.ml.posteriors import differences, situations_from, summarise, top_situations
from synergy.ml.serving import adopt_player, exposure_of, write_vectors


def _fit(prefix, situations, outcomes, worlds, kappa, positions=("TOP", "JUNGLE")):
    return {
        "prefix": prefix,
        "situations": situations,
        "outcomes": outcomes,
        "worlds": np.array(worlds, dtype=float),
        "kappa": np.array(kappa, dtype=float),
        "unit": 1.0,
        "positions": list(positions),
        "typical": np.ones(len(situations)),
    }


def test_a_situation_becomes_a_dirichlet_from_the_served_mean_and_the_chances():
    # given
    columns = ["rsp_kill_ours_far_converged", "rsp_kill_ours_far_held", "tend_follow_-m-_own", "habit_1"]
    standard = {"columns": np.array(columns), "centre": np.array([0.5, 0.5, 0.0, 0.0]), "spread": np.array([0.1, 0.1, 1.0, 1.0])}
    fits = [_fit("rsp", ["kill_ours_far"], ["converged", "held"], [[[0.4, 0.6], [0.5, 0.5]]], [8.0])]
    exposure = {"rsp_kill_ours_far": 12.0, "tend_follow_-m-_own_obs": 9.0, "tend_follow_-m-_own_exp": 4.0}

    # when
    entries = situations_from([2.0, -2.0, 1.5, 0.3], columns, exposure, "TOP", fits, {"follow": 2.0}, standard)
    shares = next(entry for entry in entries if entry["kind"] == "shares")
    ratio = next(entry for entry in entries if entry["kind"] == "ratio")
    summary = summarise(shares, np.random.default_rng(1))
    rate = summarise(ratio, np.random.default_rng(1))

    # then
    assert np.allclose(shares["alpha"], [14.0, 6.0]) and list(shares["prior"]) == [0.4, 0.6] and shares["n"] == 12.0
    assert shares["words"] == "after an ally kill from far away" and ratio["words"] == "following into fights with mid holding priority, on own half"
    assert [outcome["words"] for outcome in summary["outcomes"]] == ["converged", "held ground"]
    first = summary["outcomes"][0]
    assert first["low"] < first["mean"] == 0.7 < first["high"] and first["prior"] == 0.4
    assert rate["ratio"]["mean"] == round(11.0 / 6.0, 3) and rate["ratio"]["low"] < rate["ratio"]["mean"] < rate["ratio"]["high"] and rate["observed"] == 9.0


def test_the_strongest_situations_come_first_and_the_lane_state_collapses_to_where_the_player_stands():
    # given
    from synergy.features.priority import OUTCOMES

    columns = [*(f"prio_all_{outcome}" for outcome in OUTCOMES), "rsp_kill_ours_far_converged", "rsp_kill_ours_far_held"]
    width = len(columns)
    standard = {"columns": np.array(columns), "centre": np.full(width, 0.1), "spread": np.full(width, 0.05)}
    fits = [
        _fit("prio", ["all"], OUTCOMES, np.full((1, 1, len(OUTCOMES)), 1.0 / len(OUTCOMES)), [5.0], ["TOP"]),
        _fit("rsp", ["kill_ours_far"], ["converged", "held"], [[[0.5, 0.5]]], [8.0], ["TOP"]),
    ]
    z = np.zeros(width)
    z[0], z[-1] = 1.0, 3.0

    # when
    entries = situations_from(z, columns, {"prio_all": 300.0, "rsp_kill_ours_far": 20.0}, "TOP", fits, {}, standard)
    ranked = top_situations(entries, np.random.default_rng(0), top=2)
    lane = next(entry for entry in entries if entry["situation"] == "prio_all")

    # then
    assert [entry["situation"] for entry in ranked] == ["rsp_kill_ours_far", "prio_all"]
    assert lane["outcomes"] == ["deep_own", "own", "mid", "theirs", "deep_theirs", "off_lane", "dead"]
    assert np.isclose(lane["alpha"].sum(), 305.0) and np.isclose(lane["prior"].sum(), 1.0) and lane["words"] == "lane state"


def test_the_duo_differences_rank_shared_situations_by_how_little_the_posteriors_overlap():
    # given
    alike = {"situation": "rsp_kill_ours_far", "words": "w", "kind": "shares", "n": 30, "alpha": np.array([20.0, 10.0]), "prior": np.array([0.5, 0.5]), "outcomes": ["converged", "held"]}
    apart = {**alike, "situation": "rsp_kill_theirs_far", "alpha": np.array([27.0, 3.0])}
    left = [alike, apart, {"situation": "tend_x", "kind": "ratio"}]
    right = [{**alike, "alpha": np.array([19.0, 11.0])}, {**apart, "alpha": np.array([3.0, 27.0])}]

    # when
    found = differences(left, right, np.random.default_rng(0))

    # then
    assert [entry["situation"] for entry in found] == ["rsp_kill_theirs_far", "rsp_kill_ours_far"]
    assert found[0]["overlap"] < 0.4 < 0.8 < found[1]["overlap"]
    assert found[0]["outcomes"][0]["left"]["mean"] == 0.9 and found[0]["outcomes"][0]["right"]["mean"] == 0.1 and found[0]["outcomes"][1]["words"] == "held ground"


def test_exposure_travels_with_the_pack_and_with_adopted_players(serving_settings):
    # given
    settings = serving_settings
    pd.DataFrame({"puuid": ["a", "b"], "position": ["TOP", "JUNGLE"], "rsp_kill_ours_far": [12.0, 3.0], "tend_dive_tmb_own_obs": [2.0, 0.0]}).to_parquet(settings.processed_dir / serving.EXPOSURE_TABLE, index=False)
    columns = ["tend_dive_tmb_own", "rsp_kill_ours_near_converged"]
    adopted = {
        "columns": columns,
        "position": ["TOP"],
        "seats": np.array([4]),
        "evidence": np.array([0.5]),
        "matrix": np.array([[0.0, 1.0]], dtype=np.float32),
        "exposure": np.array([[7.0]], dtype=np.float32),
        "exposure_columns": ["rsp_kill_ours_far"],
    }

    # when
    written = write_vectors(settings.model_dir, settings.processed_dir)
    pack = serving._read_vectors(written)
    adopt_player(settings, {"puuid": "new", "champions": {}}, adopted)

    # then
    assert exposure_of(pack, "a", "TOP") == {"rsp_kill_ours_far": 12.0, "tend_dive_tmb_own_obs": 2.0}
    assert exposure_of(pack, "c", "MIDDLE") == {"rsp_kill_ours_far": 0.0, "tend_dive_tmb_own_obs": 0.0}
    assert exposure_of(serving._vectors(settings, columns), "new", "TOP") == {"rsp_kill_ours_far": 7.0}


def test_the_ranking_prefers_posteriors_that_clearly_leave_the_prior_over_single_events_on_a_rigid_cell():
    # given
    from synergy.ml.posteriors import evidence

    rigid = {"situation": "tend_follow_-m-_own", "kind": "ratio", "observed": 1.0, "expected": 0.0, "prior": 10000.0}
    loose = {"situation": "tend_greed_punished_t-b_away", "kind": "ratio", "observed": 12.0, "expected": 7.4, "prior": 16.8}
    shares = {"situation": "rsp_kill_ours_far", "kind": "shares", "n": 111.0, "alpha": np.array([50.0, 7.0, 54.0]), "prior": np.array([0.39, 0.09, 0.52]), "outcomes": ["died", "fought", "absent"]}

    # when
    rng = np.random.default_rng(3)
    scores = {entry["situation"]: evidence(entry, rng) for entry in (rigid, loose, shares)}

    # then
    assert scores["tend_follow_-m-_own"] < 0.2 < scores["tend_greed_punished_t-b_away"] < scores["rsp_kill_ours_far"]
