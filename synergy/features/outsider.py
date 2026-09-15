import gzip
import json

import numpy as np
import pandas as pd

from ..cache import get_cache
from ..config import Settings, get_settings
from ..ingest.store import Store
from ..ingest.window import truncate_timeline
from .early import early_rows
from .excursion import excursion_rows
from .player import feature_columns
from .timeline import ParsedTimeline
from .wave import wave_rows

GRID = np.linspace(0.0, 1.0, 201)
NAMESPACE = "outsider"
QUANTILES = "quantiles"


def corpus_quantiles(settings: Settings | None = None, refresh: bool = False) -> dict:
    settings = settings or get_settings()
    cache = get_cache(settings)
    if not refresh:
        cached = cache.get(QUANTILES, "participations")
        if cached:
            return cached
    frame = pd.read_parquet(settings.processed_dir / "participations.parquet")
    columns = feature_columns(frame)
    payload = {"columns": columns, "rows": int(len(frame)), "grid": GRID.tolist(), "values": {}}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        payload["values"][column] = np.quantile(values, GRID).tolist()
    cache.set(QUANTILES, "participations", payload)
    return payload


def percentile(payload: dict, column: str, value: float) -> float | None:
    table = payload["values"].get(column)
    if table is None:
        return None
    return float(np.interp(value, table, payload["grid"]) * 100.0)


def _puuids(store: Store, name: str, tag: str) -> set[str]:
    with store.conn.cursor() as cursor:
        cursor.execute(
            "SELECT puuid FROM players"
            " WHERE lower(game_name) = lower(%s) AND lower(tag_line) = lower(%s)"
            " UNION SELECT puuid FROM player_names"
            " WHERE lower(game_name) = lower(%s) AND lower(tag_line) = lower(%s)",
            (name, tag, name, tag),
        )
        return {row["puuid"] for row in cursor.fetchall()}


def _match_ids(store: Store, puuids: set[str]) -> list[str]:
    if not puuids:
        return []
    with store.conn.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT match_id FROM participations WHERE puuid = ANY(%s)",
            (list(puuids),),
        )
        return [row["match_id"] for row in cursor.fetchall()]


def _rows_for(settings: Settings, match_ids: list[str], puuids: set[str]) -> list[dict]:
    out = []
    for match_id in match_ids:
        try:
            with gzip.open(settings.match_dir / f"{match_id}.json.gz") as handle:
                match = json.load(handle)
            with gzip.open(settings.timeline_dir / f"{match_id}.json.gz") as handle:
                window = truncate_timeline(json.load(handle))
        except (FileNotFoundError, OSError):
            continue
        parsed = ParsedTimeline(match, window)
        merged = {row["puuid"]: dict(row) for row in early_rows(match, window, parsed=parsed)}
        for source in (
            wave_rows(match, window, parsed=parsed),
            excursion_rows(match, window, parsed=parsed),
        ):
            for row in source:
                merged.setdefault(row["puuid"], {}).update(
                    {k: v for k, v in row.items() if k not in ("match_id", "puuid")}
                )
        teams = {p["puuid"]: int(p.get("teamId", 0)) for p in match["info"]["participants"]}
        roles = {
            p["puuid"]: (p.get("teamPosition") or p.get("individualPosition") or "").upper()
            for p in match["info"]["participants"]
        }
        champions = {p["puuid"]: p.get("championName", "") for p in match["info"]["participants"]}
        wins = {p["puuid"]: bool(p.get("win")) for p in match["info"]["participants"]}
        for puuid, row in merged.items():
            if puuid in puuids:
                row["match_id"] = match_id
                row["team_id"] = teams.get(puuid, 0)
                row["position"] = roles.get(puuid, "")
                row["champion_name"] = champions.get(puuid, "")
                row["win"] = int(wins.get(puuid, False))
                out.append(row)
    return out


def outsider_profile(
    riot_id: str, settings: Settings | None = None, refresh: bool = False
) -> dict:
    settings = settings or get_settings()
    cache = get_cache(settings)
    key = riot_id.lower()
    if not refresh:
        cached = cache.get(NAMESPACE, key)
        if cached:
            return cached
    name, _, tag = riot_id.partition("#")
    store = Store(settings)
    try:
        puuids = _puuids(store, name, tag)
        match_ids = _match_ids(store, puuids)
    finally:
        store.close()
    rows = _rows_for(settings, match_ids, puuids)
    if not rows:
        payload = {"riot_id": riot_id, "games": 0, "features": {}}
        cache.set(NAMESPACE, key, payload)
        return payload
    frame = pd.DataFrame(rows)
    quantiles = corpus_quantiles(settings)
    features = {}
    for column in quantiles["columns"]:
        if column not in frame.columns:
            continue
        value = float(frame[column].mean())
        if value != value:
            continue
        features[column] = {
            "value": round(value, 4),
            "percentile": round(percentile(quantiles, column, value) or 0.0, 1),
        }
    payload = {
        "riot_id": riot_id,
        "games": int(len(frame)),
        "matches": sorted(frame["match_id"].unique().tolist()),
        "sides": {row.match_id: int(row.team_id) for row in frame.itertuples()},
        "compared_against": quantiles["rows"],
        "features": features,
    }
    cache.set(NAMESPACE, key, payload)
    return payload


def outsider_pair(left: str, right: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    a = outsider_profile(left, settings)
    b = outsider_profile(right, settings)
    left_sides = a.get("sides", {})
    right_sides = b.get("sides", {})
    shared = sorted(set(a.get("matches", [])) & set(b.get("matches", [])))
    together = [m for m in shared if left_sides.get(m) == right_sides.get(m)]
    against = [m for m in shared if m not in together]
    gaps = []
    for column, entry in a.get("features", {}).items():
        other = b.get("features", {}).get(column)
        if other is None:
            continue
        gaps.append(
            {
                "feature": column,
                "left": entry["percentile"],
                "right": other["percentile"],
                "gap": round(abs(entry["percentile"] - other["percentile"]), 1),
            }
        )
    gaps.sort(key=lambda item: item["gap"], reverse=True)
    return {
        "left": a,
        "right": b,
        "shared_matches": shared,
        "as_teammates": together,
        "as_opponents": against,
        "largest_gaps": gaps[:12],
    }
