import pandas as pd

from synergy.config import Settings
from synergy.features.hinge import build_hinge, hinge_between


def _responses(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["match_id", "trigger_id", "puuid", "ours", "is_actor", "converged", "present", "nearby"],
    )


def test_a_responder_who_follows_one_actor_more_than_others_scores_above_one(tmp_path):
    # given
    rows = []
    for trigger in range(12):
        rows.append(("m", trigger, "actor_a", 1, 1, 0, 0, 0))
        rows.append(("m", trigger, "follower", 1, 0, 1, 1, 1))
        rows.append(("m", trigger, "bystander", 1, 0, 0, 0, 0))
    for trigger in range(12, 24):
        rows.append(("m", trigger, "actor_b", 1, 1, 0, 0, 0))
        rows.append(("m", trigger, "follower", 1, 0, 0, 0, 0))
        rows.append(("m", trigger, "bystander", 1, 0, 0, 0, 0))
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    _responses(rows).to_parquet(settings.processed_dir / "event_responses.parquet", index=False)

    # when
    report = build_hinge(settings)
    towards_a = hinge_between("follower", "actor_a", settings)
    towards_b = hinge_between("follower", "actor_b", settings)

    # then
    assert report["responder_actor_pairs"] == 4
    assert towards_a["left_reacts_to_right"]["converged"] > 1.0
    assert towards_b["left_reacts_to_right"]["converged"] < 1.0
    assert towards_a["left_reacts_to_right"]["triggers"] == 12
    assert towards_a["right_reacts_to_left"] is None


def test_pairs_below_the_trigger_floor_are_dropped(tmp_path):
    # given
    rows = [("m", trigger, "actor", 1, 1, 0, 0, 0) for trigger in range(3)]
    rows += [("m", trigger, "responder", 1, 0, 1, 1, 1) for trigger in range(3)]
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    _responses(rows).to_parquet(settings.processed_dir / "event_responses.parquet", index=False)

    # when
    report = build_hinge(settings)

    # then
    assert report["responder_actor_pairs"] == 0
    assert hinge_between("responder", "actor", settings)["left_reacts_to_right"] is None
