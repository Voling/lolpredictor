import numpy as np
import pandas as pd

from .propensity import _gamma_prior

MIN_CELL = 25
MIN_CHANCES = 4


def prior_table(
    frame: pd.DataFrame, outcome: str, context: list[str], min_cell: int = MIN_CELL
) -> pd.DataFrame:
    subset = frame.dropna(subset=[outcome, *context])
    if subset.empty:
        return pd.DataFrame()
    base = float(subset[outcome].mean())
    table = subset.groupby(context, observed=True)[outcome].agg(games="size", rate="mean")
    table = table[table["games"] >= min_cell].copy()
    table["se"] = np.sqrt(table["rate"] * (1.0 - table["rate"]) / table["games"])
    table["base"] = base
    table["lift"] = table["rate"] - base
    table["z"] = table["lift"] / table["se"].replace(0.0, np.nan)
    return table.sort_values("rate")


def player_lift(
    frame: pd.DataFrame,
    outcome: str,
    context: list[str],
    min_chances: int = MIN_CHANCES,
    key: str = "puuid",
) -> pd.DataFrame:
    subset = frame.dropna(subset=[outcome, *context, key]).copy()
    if subset.empty:
        return pd.DataFrame()
    priors = subset.groupby(context, observed=True)[outcome].transform("mean")
    subset["expected"] = priors
    subset["variance"] = priors * (1.0 - priors)
    totals = subset.groupby(key).agg(
        chances=(outcome, "size"),
        observed=(outcome, "sum"),
        expected=("expected", "sum"),
        variance=("variance", "sum"),
    )
    totals = totals[totals["chances"] >= min_chances]
    if totals.empty:
        return totals
    strength = _gamma_prior(
        totals["observed"].to_numpy(), totals["expected"].to_numpy(), totals["variance"].to_numpy()
    )
    totals["lift"] = np.log(
        (totals["observed"] + strength) / (totals["expected"] + strength)
    )
    totals["rate"] = totals["observed"] / totals["chances"]
    totals["prior_rate"] = totals["expected"] / totals["chances"]
    totals["prior_strength"] = round(float(strength), 2)
    totals["dispersion"] = round(float(1.0 / strength), 5)
    return totals.sort_values("lift")


def separable(frame: pd.DataFrame, outcome: str, context: list[str], **kwargs) -> dict:
    lifts = player_lift(frame, outcome, context, **kwargs)
    if lifts.empty:
        return {"players": 0, "dispersion": 0.0, "separable": False}
    dispersion = float(lifts["dispersion"].iloc[0])
    return {
        "players": int(len(lifts)),
        "base_rate": round(float(frame[outcome].mean()), 4),
        "dispersion": dispersion,
        "separable": dispersion > 0.01,
    }


def describe(table: pd.DataFrame, labels: dict[tuple, str] | None = None) -> list[str]:
    lines = []
    for index, row in table.iterrows():
        key = index if isinstance(index, tuple) else (index,)
        name = (labels or {}).get(key) or " and ".join(str(part) for part in key)
        flag = "  *" if abs(row["z"]) >= 2 else ""
        lines.append(
            "  GIVEN %-34s %5.1f%%  +/- %.1f   n=%-6d %+5.1f pts%s"
            % (name, 100 * row["rate"], 100 * row["se"], row["games"], 100 * row["lift"], flag)
        )
    return lines
