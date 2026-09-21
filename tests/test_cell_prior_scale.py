import numpy as np
import pandas as pd

from synergy.features.cells import cell_block


def _counts():
    rows = []
    for match in range(60):
        for puuid, outcome in (("a", "hold"), ("b", "leave")):
            rows.append({"match_id": f"m{match}", "puuid": puuid, "situation": "s", "outcome": outcome, "count": 3.0})
    return pd.DataFrame(rows)


def _seats():
    return pd.DataFrame(
        [{"match_id": f"m{match}", "puuid": puuid, "position": "TOP"} for match in range(60) for puuid in ("a", "b")]
    )


def test_a_larger_prior_scale_pulls_every_cell_toward_the_position_world():
    # given
    counts, seats = _counts(), _seats()

    # when
    plain, report = cell_block(counts, seats, ["s"], ["hold", "leave"], "x")
    pulled, pulled_report = cell_block(counts, seats, ["s"], ["hold", "leave"], "x", scale=4.0)

    # then
    world = np.array(report["s"]["world"]["TOP"])
    own = plain.loc[plain.puuid == "a", ["x_s_hold", "x_s_leave"]].to_numpy()[0]
    shrunk = pulled.loc[pulled.puuid == "a", ["x_s_hold", "x_s_leave"]].to_numpy()[0]
    assert np.abs(shrunk - world).sum() < np.abs(own - world).sum()
    assert np.isclose(pulled_report["s"]["kappa"], 4.0 * report["s"]["kappa"], rtol=1e-6)
