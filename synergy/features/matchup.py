import numpy as np
import pandas as pd

MATCHUP_SHRINKAGE = 8.0
CHAMPION_SHRINKAGE = 12.0
COUNTER_STATES = ("countered", "even", "favoured")
LANE_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def _loo(frame: pd.DataFrame, keys: list[str], value: str, shrinkage: float) -> pd.Series:
    grouped = frame.groupby(keys, observed=True)[value]
    total = grouped.transform("sum")
    count = grouped.transform("size")
    return (total - frame[value]) / (count - 1.0 + shrinkage)


def lane_pairs(participations: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_id", "team_id", "puuid", "position", "champion_name", "e_gold_at_15"]
    frame = participations[columns].dropna(subset=["e_gold_at_15"])
    frame = frame[frame["position"].isin(LANE_ROLES)]
    merged = frame.merge(frame, on=["match_id", "position"], suffixes=("", "_enemy"))
    merged = merged[merged["team_id"] != merged["team_id_enemy"]].copy()
    merged["lane_gold"] = merged["e_gold_at_15"] - merged["e_gold_at_15_enemy"]
    return merged.reset_index(drop=True)


def matchup_edges(participations: pd.DataFrame) -> pd.DataFrame:
    pairs = lane_pairs(participations)
    if pairs.empty:
        return pairs
    pairs["centred"] = pairs["lane_gold"] - pairs.groupby("position")["lane_gold"].transform("mean")
    pairs["own_effect"] = _loo(pairs, ["position", "champion_name"], "centred", CHAMPION_SHRINKAGE)
    pairs["enemy_effect"] = _loo(
        pairs, ["position", "champion_name_enemy"], "centred", CHAMPION_SHRINKAGE
    )
    pairs["residual"] = pairs["centred"] - pairs["own_effect"] - pairs["enemy_effect"]
    pairs["edge"] = _loo(
        pairs, ["position", "champion_name", "champion_name_enemy"], "residual", MATCHUP_SHRINKAGE
    )
    pairs["matchup_games"] = pairs.groupby(
        ["position", "champion_name", "champion_name_enemy"], observed=True
    )["residual"].transform("size")
    pairs["champion_edge"] = pairs["own_effect"] + pairs["enemy_effect"]
    pairs["total_edge"] = pairs["champion_edge"] + pairs["edge"]
    return pairs


def counter_state(participations: pd.DataFrame, column: str = "total_edge") -> pd.DataFrame:
    pairs = matchup_edges(participations)
    if pairs.empty:
        return pd.DataFrame(columns=["match_id", "puuid", "position", "team_id", column, "lane_state"])
    labels = []
    for _, group in pairs.groupby("position", observed=True):
        cuts = group[column].quantile([1 / 3, 2 / 3]).to_numpy()
        labels.append(
            pd.Series(
                np.where(group[column] <= cuts[0], "countered",
                         np.where(group[column] >= cuts[1], "favoured", "even")),
                index=group.index,
            )
        )
    pairs["lane_state"] = pd.concat(labels).reindex(pairs.index)
    keep = [
        "match_id", "puuid", "team_id", "position", "champion_name", "champion_name_enemy",
        "lane_gold", "champion_edge", "edge", "total_edge", "matchup_games", "lane_state",
    ]
    return pairs[keep]


def with_ally_state(states: pd.DataFrame, ally_role: str = "JUNGLE") -> pd.DataFrame:
    ally = states[states["position"] == ally_role][
        ["match_id", "team_id", "champion_name", "total_edge", "lane_state"]
    ].rename(
        columns={
            "champion_name": f"{ally_role.lower()}_champion",
            "total_edge": f"{ally_role.lower()}_edge",
            "lane_state": f"{ally_role.lower()}_state",
        }
    )
    merged = states.merge(ally, on=["match_id", "team_id"], how="inner")
    return merged[merged["position"] != ally_role].reset_index(drop=True)


def matchup_table(participations: pd.DataFrame, position: str, min_games: int = 6) -> pd.DataFrame:
    pairs = matchup_edges(participations)
    pairs = pairs[pairs["position"] == position]
    table = pairs.groupby(["champion_name", "champion_name_enemy"], observed=True).agg(
        games=("residual", "size"), lane_gold=("lane_gold", "mean"), pair_effect=("residual", "mean")
    )
    return table[table["games"] >= min_games].sort_values("pair_effect")
