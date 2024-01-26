import pandas as pd

from ..config import Settings, get_settings
from .conditional import describe, player_lift, prior_table, separable
from .matchup import counter_state, with_ally_state
from .propensity import KINDS
from .wave import WAVE_FEATURE_COLUMNS

BEHAVIOURS = {
    "w_pushed_by_4": "gets the wave pushed in by 4:00",
    "w_pushed_at_all": "pushes at any point before 15:00",
    "w_sustained_defensive": "sits under tower three minutes straight",
}
CELL_LABELS = {
    ("even", "even"): "neither lane countered",
    ("countered", "even"): "own lane countered",
    ("even", "countered"): "jungler countered",
    ("countered", "countered"): "both countered",
    ("favoured", "favoured"): "both favoured",
}


def _read(settings: Settings, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(settings.processed_dir / f"{name}.parquet", columns=columns)


def lane_frame(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    participations = _read(
        settings,
        "participations",
        ["match_id", "team_id", "puuid", "position", "champion_name", "e_gold_at_15"],
    )
    waves = _read(settings, "waves", ["match_id", "puuid", *WAVE_FEATURE_COLUMNS])
    joined = with_ally_state(counter_state(participations))
    return joined.merge(waves, on=["match_id", "puuid"], how="inner"), participations


def push_priors(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    frame, _ = lane_frame(settings)
    out: dict = {"lanes": int(len(frame)), "matches": int(frame["match_id"].nunique()), "behaviours": {}}
    for outcome, label in BEHAVIOURS.items():
        if outcome not in frame.columns:
            continue
        table = prior_table(frame, outcome, ["lane_state", "jungle_state"])
        out["behaviours"][outcome] = {
            "label": label,
            "base_rate": round(float(frame[outcome].mean()), 4),
            "cells": table.reset_index().to_dict(orient="records"),
            "player_variation": separable(frame, outcome, ["lane_state", "jungle_state"]),
        }
    return out


def propensity_priors(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    participations = _read(
        settings,
        "participations",
        ["match_id", "team_id", "puuid", "position", "champion_name", "e_gold_at_15"],
    )
    states = counter_state(participations)[["match_id", "puuid", "lane_state"]]
    opportunities = _read(settings, "opportunities", ["match_id", "puuid", "kind", "outcome"])
    opportunities = opportunities.merge(states, on=["match_id", "puuid"], how="inner")
    out: dict = {}
    for kind in KINDS:
        subset = opportunities[opportunities["kind"] == kind]
        if len(subset) < 500:
            continue
        table = prior_table(subset, "outcome", ["lane_state"])
        out[kind] = {
            "base_rate": round(float(subset["outcome"].mean()), 4),
            "chances": int(len(subset)),
            "cells": table.reset_index().to_dict(orient="records"),
        }
    return out


def render(payload: dict) -> str:
    lines = ["%d lanes over %d matches" % (payload["lanes"], payload["matches"]), ""]
    for entry in payload["behaviours"].values():
        lines.append("%s   base rate %.1f%%" % (entry["label"], 100 * entry["base_rate"]))
        table = pd.DataFrame(entry["cells"]).set_index(["lane_state", "jungle_state"])
        lines.extend(describe(table, CELL_LABELS))
        variation = entry["player_variation"]
        lines.append(
            "  player to player spread beyond the prior: dispersion %.4f over %d players  %s"
            % (variation["dispersion"], variation["players"],
               "real trait" if variation["separable"] else "not separable from noise")
        )
        lines.append("")
    return "\n".join(lines)


def top_players(
    outcome: str = "w_pushed_by_4", limit: int = 10, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    frame, _ = lane_frame(settings)
    lifts = player_lift(frame, outcome, ["lane_state", "jungle_state"], min_chances=8)
    if lifts.empty:
        return lifts
    names = _read(settings, "participations", ["puuid", "game_name", "tag_line"]).drop_duplicates(
        "puuid"
    )
    lifts = lifts.join(names.set_index("puuid"))
    columns = ["game_name", "tag_line", "chances", "rate", "prior_rate", "lift"]
    return pd.concat([lifts.tail(limit).iloc[::-1], lifts.head(limit)])[columns]
