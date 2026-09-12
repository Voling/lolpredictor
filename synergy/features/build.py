import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from itertools import combinations

import pandas as pd

from ..chunks import ChunkWriter, merge
from ..config import Settings, get_settings
from ..ingest.store import Store
from ..ingest.window import truncate_timeline
from .advantage import fit_evaluation, state_rows
from .complement import complement_table
from .coupling import couple, coupling_table
from .detection import awareness, detection_rows
from .early import early_rows
from .families import family_rows
from .player import _refresh_axes, feature_columns
from .policy import policy_rows
from .policyvec import fit_policy_vectors
from .propensity import opportunity_rows
from .match import participant_rows
from .tempo import tempo_rows
from .timeline import ParsedTimeline, pair_rows
from .champions import champion_profiles
from .duo import fit_duo_effect
from .traits import fit_traits
from .valuesurface import fit_value_model, fit_value_surface
from .dyad import dyad_rows
from .extra import extra_rows
from .jungle import jungle_openings
from .events import event_response_rows
from .excursion import excursion_rows
from .objectives import objective_rows
from .wards import ward_rows
from .wave import response_rows, wave_rows

logger = logging.getLogger(__name__)

COUPLING_SAMPLE = 4000

STREAMED_TABLES = {
    "opportunities": opportunity_rows,
    "responses": response_rows,
    "dyads": dyad_rows,
    "jungle_openings": jungle_openings,
    "wards": ward_rows,
    "objectives": objective_rows,
    "event_responses": event_response_rows,
    "policy": policy_rows,
    "tempo": tempo_rows,
    "families": family_rows,
    "detection": detection_rows,
    "game_states": state_rows,
}
FILENAMES = {
    "participations": "participations.parquet",
    "pairs": "pair_observations.parquet",
    "opportunities": "opportunities.parquet",
    "waves": "waves.parquet",
    "responses": "responses.parquet",
    "dyads": "dyads.parquet",
    "jungle_openings": "jungle_openings.parquet",
    "wards": "wards.parquet",
    "objectives": "objectives.parquet",
    "event_responses": "event_responses.parquet",
    "policy": "policy.parquet",
    "tempo": "tempo.parquet",
    "families": "families.parquet",
    "detection": "detection.parquet",
    "game_states": "game_states.parquet",
    "excursions": "excursions.parquet",
}


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


def _extract(settings: Settings, match_ids: list[str], shard: int) -> dict:
    store = Store(settings)
    shards = settings.processed_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    names = tuple(STREAMED_TABLES) + ("waves", "participations", "pairs")
    writers = {n: ChunkWriter(shards / f"{n}.{shard:03d}.parquet") for n in names}
    try:
        for match_id in match_ids:
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
                parsed = ParsedTimeline(match, window)
                by_puuid = {row["puuid"]: row for row in early_rows(match, window, parsed=parsed)}
                wave_by_puuid = {
                    row["puuid"]: row for row in wave_rows(match, window, parsed=parsed)
                }
                for source in (
                    extra_rows(match, window, parsed=parsed),
                    excursion_rows(match, window, parsed=parsed),
                ):
                    for row in source:
                        target = by_puuid.setdefault(row["puuid"], row)
                        if target is not row:
                            target.update(
                                {k: v for k, v in row.items() if k not in ("match_id", "puuid")}
                            )
                for puuid, wave in wave_by_puuid.items():
                    target = by_puuid.setdefault(puuid, {"match_id": match_id, "puuid": puuid})
                    target.update({k: v for k, v in wave.items()
                                   if k not in ("match_id", "puuid", "role")})
                for row in rows:
                    extra = by_puuid.get(row["puuid"])
                    if extra:
                        row.update({k: v for k, v in extra.items() if k not in ("match_id", "puuid")})
                writers["pairs"].add(pair_rows(match, window))
                writers["waves"].add(list(wave_by_puuid.values()))
                for name, extract in STREAMED_TABLES.items():
                    writers[name].add(extract(match, window, parsed=parsed))
            else:
                writers["pairs"].add(_bare_pair_rows(match))
            writers["participations"].add(rows)
        return {name: writer.close() for name, writer in writers.items()}
    finally:
        store.close()


def _merge(settings: Settings, name: str, shards: int) -> int:
    directory = settings.processed_dir / "shards"
    parts = [directory / f"{name}.{index:03d}.parquet" for index in range(shards)]
    rows = merge(parts, settings.processed_dir / FILENAMES[name])
    for part in parts:
        part.unlink(missing_ok=True)
    return rows


def build_tables(
    settings: Settings | None = None,
    store: Store | None = None,
    reports: bool = True,
    workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    settings = settings or get_settings()
    owned = store is None
    store = store or Store(settings)
    try:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        allowed = qualifying_matches(store, settings)
        wanted = [m for m in store.match_ids() if allowed is None or m in allowed]
        workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) // 2))
        chunks = _split(wanted, workers)
        logger.info("extracting %s matches across %s workers", len(wanted), len(chunks))
        if len(chunks) == 1:
            results = [_extract(settings, chunks[0], 0)]
        else:
            with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
                results = list(
                    pool.map(_extract, repeat(settings), chunks, range(len(chunks)))
                )
        counts = {name: _merge(settings, name, len(chunks)) for name in results[0]}
        participations = pd.read_parquet(settings.processed_dir / FILENAMES["participations"])
        pairs = pd.read_parquet(settings.processed_dir / FILENAMES["pairs"])
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
            participations.to_parquet(
                settings.processed_dir / FILENAMES["participations"], index=False
            )
            fit_traits(participations, feature_columns(participations), settings)
            fit_duo_effect(participations, settings=settings)
            _refresh_axes()
            champion_profiles(participations, settings)
        if reports:
            fit_reports(settings, counts)
        logger.info("built %s participations and %s pair rows", len(participations), len(pairs))
        return {
            **{name: value for name, value in counts.items()},
            "participations": participations,
            "pairs": pairs,
        }
    finally:
        if owned:
            store.close()


def _split(items: list[str], parts: int) -> list[list[str]]:
    if not items:
        return [[]]
    parts = max(1, min(parts, len(items)))
    size = (len(items) + parts - 1) // parts
    return [items[index : index + size] for index in range(0, len(items), size)]


def fit_reports(settings: Settings, counts: dict[str, int]) -> dict:
    reports: dict[str, dict] = {}
    if counts.get("policy"):
        policy = pd.read_parquet(
            settings.processed_dir / FILENAMES["policy"], columns=["puuid", "state", "action"]
        )
        _, report = fit_policy_vectors(policy, settings)
        reports["policy_vectors"] = report
        del policy
        reports["coupling"] = _coupling_report(settings)
    if counts.get("game_states"):
        states = pd.read_parquet(settings.processed_dir / FILENAMES["game_states"])
        reports["evaluation"] = fit_evaluation(states, settings)
        if counts.get("policy"):
            full = pd.read_parquet(
                settings.processed_dir / FILENAMES["policy"],
                columns=["match_id", "team_id", "minute", "puuid", "role", "state", "action"],
            )
            surface = fit_value_surface(full, states, settings)
            reports["value_surface"] = {"cells": int(len(surface))}
            reports["value_model"] = fit_value_model(full, states, settings)
            pairs = pd.read_parquet(
                settings.processed_dir / FILENAMES["pairs"], columns=["puuid_a", "puuid_b"]
            )
            table = complement_table(full, pairs, settings)
            reports["complement"] = {"pairs": int(len(table))}
            del full, pairs, table, surface
        del states
    if counts.get("detection"):
        detection = pd.read_parquet(
            settings.processed_dir / FILENAMES["detection"],
            columns=["puuid", "role", "pressure", "reacted"],
        )
        table = awareness(detection)
        reports["awareness"] = {
            "players": int(len(table)),
            "d_prime_median": round(float(table["d_prime"].median()), 4) if len(table) else None,
            "bias_median": round(float(table["bias"].median()), 4) if len(table) else None,
        }
        _write_report(settings, "awareness.json", reports["awareness"])
        del detection, table
    return reports


def _coupling_report(settings: Settings) -> dict:
    policy = pd.read_parquet(
        settings.processed_dir / FILENAMES["policy"],
        columns=["match_id", "team_id", "minute", "puuid", "role", "action", "state"],
    )
    matches = policy["match_id"].drop_duplicates()
    sample = matches.sample(min(COUPLING_SAMPLE, len(matches)), random_state=0)
    policy = policy[policy["match_id"].isin(set(sample))]
    table = coupling_table(couple(policy, lag=1))
    del policy
    payload = {
        "matches": int(len(sample)),
        "role_pairs": table.to_dict(orient="records") if not table.empty else [],
    }
    _write_report(settings, "coupling.json", payload)
    return payload


def _write_report(settings: Settings, name: str, payload: dict) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_dir / name, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_tables(
    settings: Settings | None = None, names: tuple[str, ...] | None = None
) -> dict[str, pd.DataFrame]:
    settings = settings or get_settings()
    wanted = names if names is not None else tuple(FILENAMES)
    out = {}
    for name in wanted:
        path = settings.processed_dir / FILENAMES[name]
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out
