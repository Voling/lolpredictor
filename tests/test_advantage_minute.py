import json

import numpy as np
import pandas as pd

from synergy.config import Settings
from synergy.ml.gold import assemble_states, objective_prices
from synergy.ml.serving import target_minute


def test_states_carry_running_objective_counts_from_the_blue_side_and_mirror_for_red():
    # given
    frames = pd.DataFrame(
        [
            {"match_id": "m", "team_id": team, "minute": minute, "gold": gold, "xp": 1000.0, "cs": cs, "seats": 5}
            for minute in (0, 1, 2)
            for team, gold, cs in ((100, 2500.0 + 1000.0 * minute, 10.0 * minute), (200, 2500.0, 0.0))
        ]
    )
    events = pd.DataFrame(
        [
            {"match_id": "m", "team_id": 100, "minute": 1, "type": "ELITE_MONSTER_KILL", "monster_type": "RIFTHERALD", "n": 1},
            {"match_id": "m", "team_id": 200, "minute": 2, "type": "ELITE_MONSTER_KILL", "monster_type": "DRAGON", "n": 1},
            {"match_id": "m", "team_id": 100, "minute": 2, "type": "CHAMPION_KILL", "monster_type": None, "n": 2},
            {"match_id": "m", "team_id": 100, "minute": 2, "type": "ELITE_MONSTER_KILL", "monster_type": "BARON_NASHOR", "n": 1},
        ]
    )
    wins = pd.DataFrame({"match_id": ["m", "m"], "team_id": [100, 200], "win": [True, False]})

    # when
    states = assemble_states(frames, events, wins).set_index(["team_id", "minute"])

    # then
    assert states.loc[(100, 2), "gold_diff"] == 2.0 and states.loc[(100, 2), "cs_diff"] == 20.0
    assert states.loc[(100, 1), "herald_diff"] == 1.0 and states.loc[(100, 2), "herald_diff"] == 1.0
    assert states.loc[(100, 2), "dragon_diff"] == -1.0 and states.loc[(100, 2), "kill_diff"] == 2.0
    assert states.loc[(200, 2), "dragon_diff"] == 1.0 and states.loc[(200, 2), "herald_diff"] == -1.0
    assert states.loc[(100, 2), "win"] == 1 and states.loc[(200, 2), "win"] == 0
    assert "baron_diff" not in states.columns


def test_objective_prices_follow_the_target_minute_and_price_the_herald(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    weights = {name: 0.0 for name in ("gold_diff", "xp_diff", "cs_diff", "kill_diff", "plate_diff", "turret_diff", "dragon_diff", "grub_diff", "herald_diff")}
    weights.update({f"{name}_x_minute": 0.0 for name in list(weights)})
    weights.update({"gold_diff": 0.2, "gold_diff_x_minute": 0.1, "herald_diff": 0.1, "herald_diff_x_minute": 0.15, "minute": 0.0})
    (settings.processed_dir / "evaluation.json").write_text(json.dumps({"weights": weights, "intercept": 0.0}), encoding="utf-8")

    # when
    at_fifteen = objective_prices(settings, 15)
    at_twenty = objective_prices(settings, 20)

    # then
    assert np.isclose(at_fifteen["RIFTHERALD"], 0.25 / 0.3 * 1000.0)
    assert np.isclose(at_twenty["RIFTHERALD"], (0.1 + 0.15 * 20 / 15) / (0.2 + 0.1 * 20 / 15) * 1000.0)
    assert at_twenty["DRAGON"] == 0.0


def test_served_artifacts_name_their_target_minute():
    # given
    stamped, labelled, bare = {"minute": np.array(20)}, {"units": np.array("gold at 25")}, {}

    # when
    found = [target_minute(stamped), target_minute(labelled), target_minute(bare)]

    # then
    assert found == [20, 25, 15]
