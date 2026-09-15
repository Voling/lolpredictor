from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from ..config import get_settings
from ..features.player import STYLE_AXES, normaliser, participation_styles
from ..features.complement import COMPLEMENT_COLUMNS, load_complement
from ..features.dyad import DYAD_FEATURE_COLUMNS
from ..features.habit import HABIT_COLUMNS, load_habits
from ..features.orphans import ORPHAN_COLUMNS, load_orphan_features
from ..features.tendency import TENDENCY_COLUMNS, load_tendencies
from ..features.timeline import PAIR_TIMELINE_COLUMNS
from .embedding import EMBED_COLUMNS, load_embedding
from .movement import MOVEMENT_COLUMNS, load_movement

OBSERVED_GAMES = 5
STYLE_NAMES: list[str] = []
CROSS_TERMS: list[tuple[str, str]] = []
CROSS_COLUMNS: list[str] = []
DIFF_COLUMNS: list[str] = []


def refresh_columns() -> None:
    STYLE_NAMES[:] = list(STYLE_AXES)
    CROSS_TERMS[:] = list(combinations_with_replacement(STYLE_NAMES, 2))
    CROSS_COLUMNS[:] = [f"x_{left}_{right}" for left, right in CROSS_TERMS]
    DIFF_COLUMNS[:] = [f"diff_{name}" for name in STYLE_NAMES]
    HISTORY_COLUMNS[:] = [
        "hist_present",
        "hist_games_log",
        *[f"hist_{column}" for column in PAIR_HISTORY_SOURCE],
    ]
    BEHAVIOUR_COLUMNS[:] = [f"hist_{column}" for column in PAIR_HISTORY_SOURCE]
    PHI_COLUMNS[:] = CROSS_COLUMNS + DIFF_COLUMNS + BEHAVIOUR_COLUMNS
    STYLE_SUM_COLUMNS[:] = slot_columns([f"loo_style_{name}" for name in STYLE_NAMES])
    CONTROL_COLUMNS[:] = (
        STYLE_SUM_COLUMNS
        + _TEAM_CONTROLS
        + MOVEMENT_SUM_COLUMNS
        + EMBED_SUM_COLUMNS
        + ORPHAN_SUM_COLUMNS
        + TENDENCY_SUM_COLUMNS
        + HABIT_SUM_COLUMNS
    )
PAIR_HISTORY_SOURCE = PAIR_TIMELINE_COLUMNS + DYAD_FEATURE_COLUMNS
HISTORY_COLUMNS: list[str] = []
BEHAVIOUR_COLUMNS: list[str] = []
PHI_COLUMNS: list[str] = []
STYLE_SUM_COLUMNS: list[str] = []
ROLE_SLOTS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def slot_columns(columns: list[str]) -> list[str]:
    return [f"{role.lower()}_{column}" for role in ROLE_SLOTS for column in columns]


def by_role(seats: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    seated = seats[seats["position"].isin(ROLE_SLOTS)]
    wide = seated.pivot_table(
        index=["match_id", "team_id"], columns="position", values=columns, aggfunc="mean"
    )
    wide.columns = [f"{role.lower()}_{column}" for column, role in wide.columns]
    return wide.reindex(columns=slot_columns(columns))


SEAT_CONTROLS = ["lp_value", "player_games"]
_TEAM_CONTROLS = slot_columns(SEAT_CONTROLS) + ["team_lp_coverage", "team_observed"]
MOVEMENT_SUM_COLUMNS = slot_columns(MOVEMENT_COLUMNS)
EMBED_SUM_COLUMNS = slot_columns(EMBED_COLUMNS)
ORPHAN_SUM_COLUMNS = slot_columns(ORPHAN_COLUMNS)
TENDENCY_SUM_COLUMNS = slot_columns(TENDENCY_COLUMNS)
HABIT_SUM_COLUMNS = slot_columns(HABIT_COLUMNS)
CONTROL_COLUMNS: list[str] = []


def canonical_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = pairs.copy()
    swap = frame["puuid_a"] > frame["puuid_b"]
    frame.loc[swap, ["puuid_a", "puuid_b"]] = frame.loc[swap, ["puuid_b", "puuid_a"]].to_numpy()
    frame["pair_key"] = frame["puuid_a"] + "|" + frame["puuid_b"]
    return frame


def leave_one_out(frame: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    totals = frame.groupby(key)[columns].transform("sum")
    counts = frame.groupby(key)[key].transform("count").to_numpy().reshape(-1, 1)
    counts = np.repeat(counts, len(columns), axis=1)
    out = (totals.to_numpy() - frame[columns].to_numpy()) / np.where(counts > 1, counts - 1, np.nan)
    return pd.DataFrame(out, columns=columns, index=frame.index)


def player_context(
    participations: pd.DataFrame, stats: dict | None = None, shrinkage_k: float | None = None
) -> tuple[pd.DataFrame, dict]:
    stats = stats or normaliser(participations)
    k = get_settings().style_shrinkage_k if shrinkage_k is None else shrinkage_k
    styles = participation_styles(participations, stats)
    style_columns = [f"style_{name}" for name in STYLE_NAMES]

    grouped = styles.groupby("puuid")
    others = (grouped["puuid"].transform("count") - 1).to_numpy(dtype=float)
    weight = np.where(others > 0, others / (others + k), 0.0)
    totals = grouped[style_columns].transform("sum").to_numpy()
    excluded = totals - styles[style_columns].to_numpy()
    means = np.divide(excluded, others[:, None], out=np.zeros_like(excluded), where=others[:, None] > 0)

    frame = styles.copy()
    for index, column in enumerate(style_columns):
        frame[f"loo_{column}"] = means[:, index] * weight
    frame["player_games"] = grouped["puuid"].transform("count").to_numpy()
    frame["lp_value"] = (
        participations["lp_value"].to_numpy() if "lp_value" in participations.columns else np.nan
    )
    return frame, stats


def pair_history_features(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = canonical_pairs(pairs)
    available = [column for column in PAIR_HISTORY_SOURCE if column in frame.columns]
    loo = leave_one_out(
        frame.assign(**{c: frame[c].astype(float) for c in available}), "pair_key", available
    )
    counts = (frame.groupby("pair_key")["pair_key"].transform("count") - 1).to_numpy()
    out = pd.DataFrame(index=frame.index)
    out["hist_present"] = (counts > 0).astype(float)
    out["hist_games_log"] = np.log1p(np.maximum(counts, 0))
    for column in PAIR_HISTORY_SOURCE:
        values = loo[column].to_numpy() if column in available else np.zeros(len(frame))
        out[f"hist_{column}"] = np.nan_to_num(values)
    return out.reset_index(drop=True)


def attach_dyads(pairs: pd.DataFrame, dyads: pd.DataFrame) -> pd.DataFrame:
    if dyads is None or dyads.empty:
        return pairs
    columns = [c for c in DYAD_FEATURE_COLUMNS if c in dyads.columns]
    if not columns:
        return pairs
    left = canonical_pairs(dyads)[["match_id", "puuid_a", "puuid_b", *columns]]
    frame = canonical_pairs(pairs)
    merged = frame.merge(left, on=["match_id", "puuid_a", "puuid_b"], how="left")
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def phi_from_styles(
    left: np.ndarray, right: np.ndarray, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    refresh_columns()
    frame = pd.DataFrame(index=range(len(left)))
    index = {name: position for position, name in enumerate(STYLE_NAMES)}
    for column, (first, second) in zip(CROSS_COLUMNS, CROSS_TERMS):
        a_first, b_first = left[:, index[first]], right[:, index[first]]
        a_second, b_second = left[:, index[second]], right[:, index[second]]
        if first == second:
            frame[column] = a_first * b_first
        else:
            frame[column] = (a_first * b_second + a_second * b_first) / 2.0
    for column, name in zip(DIFF_COLUMNS, STYLE_NAMES):
        frame[column] = np.abs(left[:, index[name]] - right[:, index[name]])
    if history is None:
        for column in HISTORY_COLUMNS:
            frame[column] = 0.0
    else:
        for column in HISTORY_COLUMNS:
            frame[column] = history[column].to_numpy()
    return frame.fillna(0.0)


def build_pair_dataset(
    participations: pd.DataFrame,
    pairs: pd.DataFrame,
    stats: dict | None = None,
    shrinkage_k: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    refresh_columns()
    context, stats = player_context(participations, stats, shrinkage_k)
    context_key = context.set_index(["match_id", "puuid"])
    loo_columns = [f"loo_style_{name}" for name in STYLE_NAMES]
    carry = [*loo_columns, "lp_value", "position"]

    frame = canonical_pairs(pairs).reset_index(drop=True)
    history = pair_history_features(pairs)
    left = context_key[carry].reindex(pd.MultiIndex.from_arrays([frame["match_id"], frame["puuid_a"]]))
    right = context_key[carry].reindex(pd.MultiIndex.from_arrays([frame["match_id"], frame["puuid_b"]]))
    left.index = frame.index
    right.index = frame.index

    phi = phi_from_styles(
        left[loo_columns].to_numpy(dtype=float), right[loo_columns].to_numpy(dtype=float), history
    )
    phi.index = frame.index
    complement = load_complement()
    if not complement.empty:
        merged = frame[["puuid_a", "puuid_b"]].merge(
            complement, on=["puuid_a", "puuid_b"], how="left"
        )
        for column in COMPLEMENT_COLUMNS:
            phi[column] = merged[column].fillna(0.0).to_numpy()
    else:
        for column in COMPLEMENT_COLUMNS:
            phi[column] = 0.0
    keep = left[loo_columns].notna().all(axis=1) & right[loo_columns].notna().all(axis=1)
    features = pd.concat(
        [frame[["match_id", "team_id", "pair_key", "puuid_a", "puuid_b", "win"]], phi], axis=1
    )[keep]

    grouped = context.groupby(["match_id", "team_id"])
    controls = grouped.agg(
        team_lp_coverage=("lp_value", lambda values: values.notna().mean()),
        team_observed=("player_games", lambda values: int((values >= OBSERVED_GAMES).sum())),
    )
    controls = controls.join(by_role(context, SEAT_CONTROLS))
    controls = controls.join(by_role(context, loo_columns))
    movement = load_movement()
    if not movement.empty:
        seats = context[["match_id", "team_id", "puuid", "position"]].merge(
            movement, on=["match_id", "puuid"], how="left"
        )
        controls = controls.join(by_role(seats, MOVEMENT_COLUMNS))
    else:
        for column in MOVEMENT_SUM_COLUMNS:
            controls[column] = 0.0

    learned = load_embedding()
    if not learned.empty:
        seats = context[["match_id", "team_id", "puuid", "position"]].merge(
            learned, on=["match_id", "puuid"], how="left"
        )
        controls = controls.join(by_role(seats, EMBED_COLUMNS))
    else:
        for column in EMBED_SUM_COLUMNS:
            controls[column] = 0.0

    orphans = load_orphan_features()
    if not orphans.empty:
        seats = context[["match_id", "team_id", "puuid", "position"]].merge(
            orphans, on=["match_id", "puuid"], how="left"
        )
        controls = controls.join(by_role(seats, ORPHAN_COLUMNS))
    else:
        for column in ORPHAN_SUM_COLUMNS:
            controls[column] = 0.0

    tendencies = load_tendencies()
    if not tendencies.empty:
        seats = context[["match_id", "team_id", "puuid", "position"]].merge(
            tendencies, on=["match_id", "puuid"], how="left"
        )
        controls = controls.join(by_role(seats, TENDENCY_COLUMNS))
    else:
        for column in TENDENCY_SUM_COLUMNS:
            controls[column] = 0.0

    habits = load_habits()
    if not habits.empty:
        seats = context[["match_id", "team_id", "puuid", "position"]].merge(
            habits, on=["match_id", "puuid"], how="left"
        )
        controls = controls.join(by_role(seats, HABIT_COLUMNS))
    else:
        for column in HABIT_SUM_COLUMNS:
            controls[column] = 0.0
    for column in CONTROL_COLUMNS:
        if column in controls.columns:
            controls[column] = controls[column].fillna(controls[column].mean()).fillna(0.0)
    return features.reset_index(drop=True), controls.reset_index(), stats


def build_team_dataset(
    features: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    refresh_columns()
    grouped = features.groupby(["match_id", "team_id"])
    merged = grouped[PHI_COLUMNS].sum().join(grouped["win"].first())
    merged = merged.join(controls.set_index(["match_id", "team_id"])).dropna()
    blue = merged.xs(100, level="team_id")
    red = merged.xs(200, level="team_id")
    shared = blue.index.intersection(red.index)
    blue, red = blue.loc[shared], red.loc[shared]
    columns = PHI_COLUMNS + CONTROL_COLUMNS
    observed = (blue["team_observed"] + red["team_observed"]).to_numpy()
    return blue[columns] - red[columns], blue["win"].to_numpy().astype(int), observed


refresh_columns()
