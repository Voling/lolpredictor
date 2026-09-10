import logging
from itertools import combinations

import pandas as pd

from ..config import Settings, get_settings
from ..ingest.store import Store
from ..ingest.window import truncate_timeline
from .early import early_rows
from .propensity import opportunity_rows
from .match import participant_rows
from .timeline import pair_rows
from .champions import champion_profiles
from .duo import fit_duo_effect
from .traits import fit_traits
from .dyad import dyad_rows
from .extra import extra_rows
from .jungle import jungle_openings
from .events import event_response_rows
from .objectives import objective_rows
from .wards import ward_rows
from .wave import response_rows, wave_rows

logger = logging.getLogger(__name__)


def _bare_pair_rows(match: dict) -> list[dict]:
    match_id = match["metadata"]["matchId"]
    rows = []
    for team in (100, 200):
        members = [p for p in match["info"]["participants"] if int(p.get("teamId", 0)) == team]
        if not members:
            continue
        win = int(bool(members[0].get("win")))
        for left, right in combinations(sorted(members, key=lambda p: p["puuid"]), 2):
            rows.append(
                {
                    "match_id": match_id,
                    "puuid_a": left["puuid"],
                    "puuid_b": right["puuid"],
                    "team_id": team,
                    "win": win,
                }
            )
    return rows


def qualifying_matches(store: Store, settings: Settings) -> set[str] | None:
    if settings.min_average_lp <= 0:
        return None
    summary = pd.DataFrame(store.match_rank_summary())
    if summary.empty or summary["ranked"].sum() == 0:
        return None
    judged = summary["ranked"] >= settings.min_ranked_participants
    below = judged & (summary["average_lp"].astype(float) < settings.min_average_lp)
    stale = pd.Series(False, index=summary.index)
    if settings.season_start and "game_creation" in summary.columns:
        floor = pd.Timestamp(settings.season_start, tz="UTC")
        when = pd.to_datetime(summary["game_creation"], utc=True, errors="coerce")
        stale = when.notna() & (when < floor)
    drop = below | stale
    logger.info(
        "corpus filter: dropped %s below %s average lp, %s before %s, kept %s",
        int(below.sum()), settings.min_average_lp, int(stale.sum()),
        settings.season_start, int((~drop).sum()),
    )
    return set(summary.loc[~drop, "match_id"])


def build_tables(settings: Settings | None = None, store: Store | None = None) -> dict[str, pd.DataFrame]:
    settings = settings or get_settings()
    owned = store is None
    store = store or Store(settings)
    try:
        participation_records: list[dict] = []
        pair_records: list[dict] = []
        opportunity_records: list[dict] = []
        wave_records: list[dict] = []
        response_records: list[dict] = []
        dyad_records: list[dict] = []
        ward_records: list[dict] = []
        objective_records: list[dict] = []
        event_records: list[dict] = []
        allowed = qualifying_matches(store, settings)
        opening_records: list[dict] = []
        for match_id in store.match_ids():
            if allowed is not None and match_id not in allowed:
                continue
            try:
                match = store.load_match(match_id)
            except FileNotFoundError:
                continue
            rows = participant_rows(match)
            window = None
            try:
                if store.has_window(match_id):
                    window = store.load_window(match_id)
                elif store.has_timeline(match_id):
                    window = truncate_timeline(store.load_timeline(match_id))
            except FileNotFoundError:
                window = None
            if window is not None:
                by_puuid = {row["puuid"]: row for row in early_rows(match, window)}
                wave_by_puuid = {row["puuid"]: row for row in wave_rows(match, window)}
                for row in extra_rows(match, window):
                    target = by_puuid.setdefault(row["puuid"], row)
                    if target is not row:
                        target.update({k: v for k, v in row.items() if k not in ("match_id", "puuid")})
                for puuid, wave in wave_by_puuid.items():
                    target = by_puuid.setdefault(puuid, {"match_id": match_id, "puuid": puuid})
                    target.update({k: v for k, v in wave.items()
                                   if k not in ("match_id", "puuid", "role")})
                for row in rows:
                    extra = by_puuid.get(row["puuid"])
                    if extra:
                        row.update({k: v for k, v in extra.items() if k not in ("match_id", "puuid")})
                pair_records.extend(pair_rows(match, window))
                opportunity_records.extend(opportunity_rows(match, window))
                wave_records.extend(wave_by_puuid.values())
                response_records.extend(response_rows(match, window))
                dyad_records.extend(dyad_rows(match, window))
                ward_records.extend(ward_rows(match, window))
                opening_records.extend(jungle_openings(match, window))
                objective_records.extend(objective_rows(match, window))
                event_records.extend(event_response_rows(match, window))
            else:
                pair_records.extend(_bare_pair_rows(match))
            participation_records.extend(rows)
        participations = pd.DataFrame(participation_records)
        pairs = pd.DataFrame(pair_records)
        if not participations.empty:
            players = pd.DataFrame(store.players())
            if not players.empty:
                keep = [
                    c
                    for c in (
                        "puuid",
                        "game_name",
                        "tag_line",
                        "tier",
                        "division",
                        "lp_value",
                        "wins",
                        "losses",
                    )
                    if c in players
                ]
                participations = participations.merge(players[keep], on="puuid", how="left")
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        participations.to_parquet(settings.processed_dir / "participations.parquet", index=False)
        pairs.to_parquet(settings.processed_dir / "pair_observations.parquet", index=False)
        opportunities = pd.DataFrame(opportunity_records)
        opportunities.to_parquet(settings.processed_dir / "opportunities.parquet", index=False)
        waves = pd.DataFrame(wave_records)
        waves.to_parquet(settings.processed_dir / "waves.parquet", index=False)
        responses = pd.DataFrame(response_records)
        responses.to_parquet(settings.processed_dir / "responses.parquet", index=False)
        dyads = pd.DataFrame(dyad_records)
        dyads.to_parquet(settings.processed_dir / "dyads.parquet", index=False)
        wards = pd.DataFrame(ward_records)
        wards.to_parquet(settings.processed_dir / "wards.parquet", index=False)
        openings = pd.DataFrame(opening_records)
        openings.to_parquet(settings.processed_dir / "jungle_openings.parquet", index=False)
        objectives = pd.DataFrame(objective_records)
        objectives.to_parquet(settings.processed_dir / "objectives.parquet", index=False)
        events = pd.DataFrame(event_records)
        events.to_parquet(settings.processed_dir / "event_responses.parquet", index=False)
        if not participations.empty:
            from .player import _refresh_axes, feature_columns

            fit_traits(participations, feature_columns(participations), settings)
            fit_duo_effect(participations, settings=settings)
            _refresh_axes()
            champion_profiles(participations, settings)
        logger.info("built %s participations and %s pair rows", len(participations), len(pairs))
        return {
            "participations": participations,
            "pairs": pairs,
            "opportunities": opportunities,
            "waves": waves,
            "responses": responses,
            "dyads": dyads,
            "wards": wards,
            "jungle_openings": openings,
            "objectives": objectives,
            "event_responses": events,
        }
    finally:
        if owned:
            store.close()


def load_tables(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    settings = settings or get_settings()
    return {
        "participations": pd.read_parquet(settings.processed_dir / "participations.parquet"),
        "pairs": pd.read_parquet(settings.processed_dir / "pair_observations.parquet"),
        "opportunities": pd.read_parquet(settings.processed_dir / "opportunities.parquet"),
        "waves": pd.read_parquet(settings.processed_dir / "waves.parquet"),
        "responses": pd.read_parquet(settings.processed_dir / "responses.parquet"),
        "dyads": pd.read_parquet(settings.processed_dir / "dyads.parquet"),
        "wards": pd.read_parquet(settings.processed_dir / "wards.parquet"),
        "jungle_openings": pd.read_parquet(settings.processed_dir / "jungle_openings.parquet"),
        "objectives": pd.read_parquet(settings.processed_dir / "objectives.parquet"),
        "event_responses": pd.read_parquet(settings.processed_dir / "event_responses.parquet"),
    }
