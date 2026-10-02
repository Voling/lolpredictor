import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .features.build import FILENAMES, _extract, _merge
from .features.cells import SeatIndex, combine_shares, dense_counts
from .features.exposure import exposure_frame, seat_exposure
from .features.fits import (
    BUNDLE,
    FIT_FILES,
    PRIORITY_FIT,
    PRIORS_FILE,
    REACTION_FITS,
    TENDENCY_FIT,
    apply_cells,
    apply_tendencies,
    load_cells,
    load_tendencies,
    reaction_sources,
    save_priors,
    tendency_totals,
)
from .features.habit import HABIT_COLUMNS, SMOOTHING
from .features.policy import ACTIONS, STATES
from .features.positions import KEY
from .features.priority import PRIORITY_COLUMNS, _shard as priority_shard, priority_context_from
from .features.reaction import REACTION_COLUMNS
from .features.tendency import TENDENCY_COLUMNS
from .ingest.store import Store
from .ingest.window import truncate_timeline
from .ml.embedding import DIMS, EMBED_COLUMNS, MIN_GAMES
from .ml.movement import MIN_TRANSITIONS, MOVEMENT_COLUMNS, _count as movement_counts, signatures

logger = logging.getLogger(__name__)
EVALUATED = "evaluated"
TABLES =("participations", "opportunities", "event_responses", "objectives", "wards", "jungle_openings", "policy")
BUNDLE_FILES = (*FIT_FILES, "habit_basis.npz", "movement_basis.npz", "embedding_basis.npz", "timeline_encoder.pt", "position_sigma.json")
STEPS = ("fetching", "loading", "features", "positions", "reactions", "tendencies", "habits", "movement", "embedding", "scoring")


def write_bundle(settings: Settings | None = None, target: Path | None = None) -> Path:
    settings = settings or get_settings()
    run = settings.served_run()
    target = target or (settings.runs_dir / run / "serve" / BUNDLE if run else settings.model_dir / BUNDLE)
    target.mkdir(parents=True, exist_ok=True)
    missing = [name for name in BUNDLE_FILES if not (settings.model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"the model dir lacks {', '.join(missing)}; run the blocks and `python -m synergy fits` first")
    for name in BUNDLE_FILES:
        shutil.copy2(settings.model_dir / name, target / name)
    save_priors(target / PRIORS_FILE, load_tendencies(settings.model_dir / TENDENCY_FIT))
    with np.load(settings.model_dir / "movement_transitions.npz", allow_pickle=True) as moves:
        np.savez(target / "movement_world.npz", world=moves["world"], positions=moves["positions"], kappa=moves["kappa"])
    style = json.loads((settings.processed_dir / "basis" / "style.json").read_text(encoding="utf-8"))
    with np.load(settings.processed_dir / "basis" / "reduced.npz") as reduced:
        np.savez(target / "standardise.npz", columns=np.array(style["columns"]), centre=reduced["centre"], spread=reduced["spread"])
    return target


class Bundle:
    def __init__(self, folder: Path):
        self.folder = folder
        with np.load(folder / "standardise.npz") as data:
            self.columns = [str(name) for name in data["columns"]]
            self.centre, self.spread = data["centre"].astype(float), data["spread"].astype(float)
        self.priority = load_cells(folder / PRIORITY_FIT)
        self.reaction = [load_cells(folder / name) for name in REACTION_FITS]
        self.tendency = load_tendencies(folder / TENDENCY_FIT)
        with np.load(folder / "habit_basis.npz", allow_pickle=True) as data:
            self.habit = {name: data[name] for name in ("basis", "centre", "base", "positions")}
        with np.load(folder / "movement_world.npz", allow_pickle=True) as data:
            self.movement_world = {name: data[name] for name in ("world", "positions", "kappa")}
        with np.load(folder / "movement_basis.npz") as data:
            self.movement_basis = {name: data[name] for name in ("basis", "centre")}
        with np.load(folder / "embedding_basis.npz") as data:
            self.embedding_basis = {name: data[name] for name in ("basis", "centre")}


def habit_rows(policy: pd.DataFrame, seats: pd.DataFrame, saved: dict) -> pd.DataFrame:
    policy = policy[policy.state.isin(STATES) & policy.action.isin(ACTIONS)].merge(seats[["match_id", *KEY]], on=["match_id", "puuid"], how="inner")
    if policy.empty:
        return pd.DataFrame(columns=["match_id", "puuid", *HABIT_COLUMNS])
    cells = pd.MultiIndex.from_product([STATES, ACTIONS], names=["state", "action"])
    slot = policy.set_index(["state", "action"]).index.map({pair: index for index, pair in enumerate(cells)}).to_numpy()
    seat_codes, seat_index = pd.factorize(policy.match_id.astype(str) + "|" + policy.puuid.astype(str))
    people, person_index = pd.factorize(policy.puuid.astype(str) + "|" + policy.position.astype(str))
    width = len(cells)
    per_seat = np.zeros((len(seat_index), width))
    np.add.at(per_seat, (seat_codes, slot), 1.0)
    per_person = np.zeros((len(person_index), width))
    np.add.at(per_person, (people, slot), 1.0)
    owner = np.zeros(len(seat_index), dtype=np.int64)
    owner[seat_codes] = people
    others = per_person[owner] - per_seat
    person_position = np.array([key.rsplit("|", 1)[1] for key in person_index])
    positions = [str(name) for name in saved["positions"]]
    code = pd.Index(positions).get_indexer(person_position[owner])
    known = code >= 0
    base_of_seat = saved["base"][np.where(known, code, 0)]
    prior = SMOOTHING * base_of_seat
    shaped = others.reshape(-1, len(STATES), len(ACTIONS))
    rates = (shaped + prior) / (shaped.sum(axis=2, keepdims=True) + SMOOTHING)
    flat = (np.log(rates) - np.log(base_of_seat)).reshape(len(seat_index), width)
    enough = (others.sum(axis=1) > 0) & known
    scores = np.where(enough[:, None], (flat - saved["centre"]) @ saved["basis"].T, np.nan)
    keys = pd.Series(seat_index).str.split("|", n=1, expand=True)
    table = pd.DataFrame(scores, columns=HABIT_COLUMNS[: saved["basis"].shape[0]])
    table.insert(0, "puuid", keys[1].to_numpy())
    table.insert(0, "match_id", keys[0].to_numpy())
    return table


def movement_rows(settings: Settings, match_ids: list[str], puuid: str, world: dict, basis: dict) -> pd.DataFrame:
    players, train, test = movement_counts(settings, match_ids, 0)
    if not players:
        return pd.DataFrame(columns=[*KEY, *MOVEMENT_COLUMNS])
    every = train + test
    names = [str(name) for name in world["positions"]]
    keys = pd.Series(players).str.rsplit("|", n=1, expand=True)
    mine = (keys[0] == puuid).to_numpy()
    code = pd.Index(names).get_indexer(keys[1])
    keep = mine & (code >= 0) & (every.sum(axis=(1, 2)) >= MIN_TRANSITIONS)
    if not keep.any():
        return pd.DataFrame(columns=[*KEY, *MOVEMENT_COLUMNS])
    signature = signatures(every[keep], world["world"][code[keep]], float(world["kappa"][0])).reshape(int(keep.sum()), -1)
    scores = (signature - basis["centre"]) @ basis["basis"].T
    frame = pd.DataFrame(scores, columns=MOVEMENT_COLUMNS[: basis["basis"].shape[0]])
    frame.insert(0, "position", keys[1].to_numpy()[keep])
    frame.insert(0, "puuid", puuid)
    return frame


def embedding_rows(games: list[tuple[dict, dict]], puuid: str, encoder_path: Path, basis: dict) -> pd.DataFrame:
    import torch

    from .deep.model import TimelineEncoder
    from .deep.sequences import NO_EVENT_REGION, encode_match
    from .deep.train import infer
    from .features.regions import REGIONS

    saved = torch.load(encoder_path, map_location="cpu", weights_only=False)
    config, state = saved["config"], saved["state_dict"]
    rows = [row for match, window in games for row in encode_match(match, window, int(config["minutes"])) if row["puuid"] == puuid]
    if not rows:
        return pd.DataFrame(columns=[*KEY, *EMBED_COLUMNS])
    encoder = TimelineEncoder(
        regions=len(REGIONS),
        event_regions=NO_EVENT_REGION + 1,
        numeric=int(config["numeric"]),
        minutes=int(config["minutes"]),
        champions=int(state["champion_embedding.weight"].shape[0]),
        roles=int(state["role_embedding.weight"].shape[0]),
        dim=int(config["dim"]),
        embed_dim=int(config["embed_dim"]),
        layers=int(config["layers"]),
    )
    encoder.load_state_dict(state)
    numeric = torch.from_numpy(np.stack([row["numeric"] for row in rows])).float()
    mask = torch.from_numpy(np.stack([row["mask"] for row in rows]))
    numeric = ((numeric - torch.from_numpy(saved["numeric_mean"]).float()) / torch.from_numpy(saved["numeric_std"]).float()) * mask[..., None]
    with torch.no_grad():
        embeddings = infer(
            encoder,
            torch.from_numpy(np.stack([row["regions"] for row in rows]).astype(np.int64)),
            torch.from_numpy(np.stack([row["event_regions"] for row in rows]).astype(np.int64)),
            numeric,
            mask,
            torch.zeros(len(rows), dtype=torch.int64),
            torch.zeros(len(rows), dtype=torch.int64),
            "cpu",
        )
    frame = pd.DataFrame(embeddings, columns=[f"e{index}" for index in range(DIMS)])
    frame["position"] = [row["position"] for row in rows]
    grouped = frame.groupby("position")
    signature = grouped.mean()
    signature = signature[grouped.size() >= MIN_GAMES]
    if signature.empty:
        return pd.DataFrame(columns=[*KEY, *EMBED_COLUMNS])
    scores = (signature.to_numpy(dtype=float) - basis["centre"]) @ basis["basis"].T
    out = pd.DataFrame(scores, columns=EMBED_COLUMNS[: basis["basis"].shape[0]])
    out.insert(0, "position", signature.index.to_numpy())
    out.insert(0, "puuid", puuid)
    return out


class Evaluator:
    def __init__(self, bundle: Bundle, settings: Settings, progress=lambda step, done, total: None):
        self.bundle, self.settings, self.progress = bundle, settings, progress

    def _step(self, name: str, done: int = 0, total: int = 0) -> None:
        logger.info("evaluate: %s %s/%s", name, done, total)
        self.progress(name, done, total)

    def load(self, games) -> list[str]:
        kept = []
        store = Store(self.settings)
        try:
            for number, (match_id, match, timeline) in enumerate(games, start=1):
                if store.save_match(match, source=EVALUATED) is False:
                    continue
                store.save_timeline(match_id, timeline)
                store.save_window(match_id, truncate_timeline(timeline))
                kept.append(match_id)
                self._step("loading", number, 0)
        finally:
            store.close()
        return kept

    def run(self, puuid: str, games: list[tuple[str, dict, dict]]) -> dict:
        settings, bundle = self.settings, self.bundle
        settings.ensure_dirs()
        clock = time.time()
        self._step("loading", 0, len(games))
        match_ids = self.load(games)
        if not match_ids:
            raise ValueError("none of the games could be loaded")
        self._step("features")
        _extract(settings, match_ids, 0)
        for name in TABLES:
            _merge(settings, name, 1)
        processed = settings.processed_dir
        seats = pd.read_parquet(processed / FILENAMES["participations"], columns=["match_id", "puuid", "team_id", "position"])
        index = SeatIndex(seats)
        self._step("positions")
        track_path, counts_path = priority_shard(settings, match_ids, 0)
        with np.load(counts_path) as shard:
            counts = np.zeros((len(index), *shard["counts"].shape[1:]))
            rows = index.rows(shard["match_id"], shard["puuid"])
            counts[rows[rows >= 0]] = shard["counts"][rows >= 0]
        priority, priority_evidence = apply_cells(bundle.priority, counts, index)
        exposure_parts = [("prio", bundle.priority["situations"], seat_exposure(counts, index, bundle.priority["situations"], bundle.priority["unit"]))]
        context = priority_context_from(pd.read_parquet(track_path), seats)
        self._step("reactions")
        reaction_parts, shares = [], []
        for number, (prefix, load, situations, outcomes) in enumerate(reaction_sources(settings)):
            counts = np.zeros((len(index), len(situations), len(outcomes)))
            for frame in load():
                dense_counts(frame, index, situations, outcomes, out=counts)
            frame, evidence = apply_cells(bundle.reaction[number], counts, index)
            reaction_parts.append(frame)
            shares.append((len(situations) * len(outcomes), evidence))
            exposure_parts.append((prefix, situations, seat_exposure(counts, index, situations)))
        reaction = reaction_parts[0]
        for frame in reaction_parts[1:]:
            reaction = reaction.merge(frame, on=["match_id", "puuid"], how="outer")
        reaction_evidence = combine_shares(shares)
        self._step("tendencies")
        opportunities = pd.read_parquet(processed / FILENAMES["opportunities"])
        tendency, tendency_evidence = apply_tendencies(bundle.tendency, opportunities, context, seats[["match_id", "puuid", "position"]])
        totals = tendency_totals(bundle.tendency, lambda kind: opportunities[opportunities["kind"] == kind], context, seats[["match_id", "puuid", "position"]])
        exposure = exposure_frame(index, exposure_parts).merge(totals, on=KEY, how="left").fillna(0.0).set_index(KEY)
        self._step("habits")
        habits = habit_rows(pd.read_parquet(processed / FILENAMES["policy"], columns=["match_id", "puuid", "state", "action"]), seats, bundle.habit)
        self._step("movement")
        movement = movement_rows(settings, match_ids, puuid, bundle.movement_world, bundle.movement_basis)
        self._step("embedding")
        store = Store(settings)
        try:
            windows = [(store.load_match(match_id), store.load_window(match_id)) for match_id in match_ids]
        finally:
            store.close()
        embedding = embedding_rows(windows, puuid, bundle.folder / "timeline_encoder.pt", bundle.embedding_basis)
        self._step("scoring")
        mine = seats[seats.puuid == puuid][["match_id", "puuid", "position"]].drop_duplicates(["match_id", "puuid"])
        table = mine.copy()
        for frame in (tendency, priority, reaction, habits):
            table = table.merge(frame, on=["match_id", "puuid"], how="left")
        for frame in (movement, embedding):
            table = table.merge(frame, on=KEY, how="left")
        for column in bundle.columns:
            if column not in table.columns:
                table[column] = np.nan
        values = table[bundle.columns].to_numpy(dtype=float)
        standard = np.nan_to_num((values - bundle.centre) / bundle.spread, nan=0.0).astype(np.float32)
        positions = table["position"].to_numpy()
        names = sorted(set(positions))
        matrix = np.stack([standard[positions == name].mean(axis=0) for name in names])
        seat_counts = np.array([int((positions == name).sum()) for name in names], dtype=np.int32)
        weights = ((len(REACTION_COLUMNS), reaction_evidence), (len(PRIORITY_COLUMNS), priority_evidence), (len(TENDENCY_COLUMNS), tendency_evidence))
        evidence = combine_shares([(weight, frame) for weight, frame in weights]).set_index(KEY)["share"]
        evidence_values = np.array([float(evidence.get((puuid, name), np.nan)) for name in names], dtype=np.float32)
        exposure_columns = list(exposure.columns)
        blank = np.zeros(len(exposure_columns), dtype=np.float32)
        exposure_rows = np.stack([exposure.loc[(puuid, name)].to_numpy(dtype=np.float32) if (puuid, name) in exposure.index else blank for name in names])
        logger.info("evaluate: %s games, %s positions in %.0fs", len(match_ids), len(names), time.time() - clock)
        return {
            "puuid": puuid,
            "columns": bundle.columns,
            "position": np.array(names),
            "seats": seat_counts,
            "evidence": evidence_values,
            "matrix": matrix,
            "exposure": exposure_rows,
            "exposure_columns": exposure_columns,
            "games": len(match_ids),
            "seats_total": int(len(mine)),
        }


def profile_of(puuid: str, games: list[tuple[str, dict, dict]], account: dict, league: list[dict], result: dict) -> dict:
    champions, positions, wins, by_position, ended = {}, {}, [], {}, []
    for _, match, _ in games:
        info = match["info"]
        ended.append(int(info.get("gameEndTimestamp") or int(info.get("gameCreation", 0)) + int(info.get("gameDuration", 0))))
        for participant in info["participants"]:
            if participant["puuid"] != puuid:
                continue
            champion = participant.get("championName", "")
            champions[champion] = champions.get(champion, 0) + 1
            position = participant.get("teamPosition") or participant.get("individualPosition") or ""
            positions[position] = positions.get(position, 0) + 1
            by_position.setdefault(position, {})[champion] = by_position.get(position, {}).get(champion, 0) + 1
            wins.append(bool(participant.get("win")))
    solo = next((entry for entry in league if entry.get("queueType") == "RANKED_SOLO_5x5"), None)
    return {
        "puuid": puuid,
        "game_name": account.get("gameName"),
        "tag_line": account.get("tagLine"),
        "games": result["games"],
        "winrate": round(sum(wins) / len(wins), 4) if wins else 0.0,
        "main_position": max(positions, key=positions.get) if positions else None,
        "main_champion": max(champions, key=champions.get) if champions else None,
        "champion_pool": len(champions),
        "tier": solo.get("tier") if solo else None,
        "division": solo.get("rank") if solo else None,
        "lp_value": solo.get("leaguePoints") if solo else None,
        "latest_game": max(ended) // 1000 if ended else None,
        "positions": [str(name) for name in result["position"]],
        "seats": [int(count) for count in result["seats"]],
        "champions": by_position,
    }


def save_result(folder: Path, result: dict, profile: dict) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    vectors = folder / f"{result['puuid']}.npz"
    extra = {"exposure": result["exposure"], "exposure_columns": np.array(result["exposure_columns"], dtype=str)} if "exposure" in result else {}
    np.savez(vectors, columns=np.array(result["columns"]), position=result["position"], seats=result["seats"], evidence=result["evidence"], matrix=result["matrix"], **extra)
    summary = folder / f"{result['puuid']}.json"
    summary.write_text(json.dumps(profile), encoding="utf-8")
    return vectors, summary
