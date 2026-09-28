import asyncio
import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..ranks import pick_solo_entry, rank_fields
from ..riot.client import NotFound, RiotClient
from .lock import corpus_lock
from .store import Store, conclusive, in_season

KINDS = ("matches", "timelines")
SUFFIX = ".json.gz"
SOURCE = "contributed"


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def ids(self) -> list[str]:
        found = [{path.name[: -len(SUFFIX)] for path in (self.root / kind).glob(f"*{SUFFIX}")} for kind in KINDS]
        return sorted(found[0] & found[1])

    def read(self, kind: str, match_id: str) -> dict:
        with gzip.open(self.root / kind / f"{match_id}{SUFFIX}", "rt", encoding="utf-8") as handle:
            return json.load(handle)


class S3Store:
    def __init__(self, bucket: str, prefix: str = "", client=None):
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.bucket, self.prefix, self.client = bucket, prefix.strip("/"), client

    def _key(self, *parts: str) -> str:
        return "/".join(part for part in (self.prefix, *parts) if part)

    def _names(self, kind: str) -> set[str]:
        names = set()
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self._key(kind) + "/"):
            for item in page.get("Contents", []):
                name = item["Key"].rsplit("/", 1)[-1]
                if name.endswith(SUFFIX):
                    names.add(name[: -len(SUFFIX)])
        return names

    def ids(self) -> list[str]:
        return sorted(self._names("matches") & self._names("timelines"))

    def read(self, kind: str, match_id: str) -> dict:
        body = self.client.get_object(Bucket=self.bucket, Key=self._key(kind, f"{match_id}{SUFFIX}"))["Body"].read()
        return json.loads(gzip.decompress(body).decode("utf-8"))


def open_store(location: str):
    if not location:
        raise ValueError("Set CONTRIBUTED_STORE to an s3://bucket/prefix or a folder")
    if location.startswith("s3://"):
        bucket, _, prefix = location.removeprefix("s3://").partition("/")
        return S3Store(bucket, prefix)
    return LocalStore(Path(location))


def basic_reason(match: dict, settings: Settings) -> str | None:
    info = match.get("info", {})
    if str(info.get("platformId", "")).lower() != settings.platform.lower():
        return "another platform"
    if info.get("queueId") != settings.queue_id:
        return "another queue"
    if not in_season(".".join(str(info.get("gameVersion", "")).split(".")[:2]), settings.season):
        return "another season"
    if not conclusive(info):
        return "unfinished"
    return None


def rank_reason(players: list[str], ranks: dict[str, int], settings: Settings) -> str | None:
    known = [ranks[puuid] for puuid in players if puuid in ranks]
    if len(known) < settings.min_ranked_participants:
        return "too few ranked players"
    if sum(known) / len(known) < settings.min_average_lp:
        return f"average below {settings.min_average_lp} LP"
    return None


@dataclass
class Review:
    waiting: int = 0
    known: int = 0
    reasons: Counter = field(default_factory=Counter)
    players: dict[str, list[str]] = field(default_factory=dict)
    looked_up: dict[str, dict] = field(default_factory=dict)

    @property
    def qualified(self) -> list[str]:
        return list(self.players)

    def summary(self) -> dict:
        return {"waiting": self.waiting, "in corpus": self.known, "qualified": len(self.players), **dict(self.reasons)}


def gather(settings: Settings, source, store: Store, lookup) -> Review:
    review = Review()
    listed = source.ids()
    present = store.known_matches(listed)
    review.waiting, review.known = len(listed), len(present)
    candidates = {}
    for match_id in listed:
        if match_id in present:
            continue
        match = source.read("matches", match_id)
        reason = basic_reason(match, settings)
        if reason:
            review.reasons[reason] += 1
        else:
            candidates[match_id] = [p["puuid"] for p in match["info"]["participants"] if p.get("puuid")]
    everyone = sorted({puuid for players in candidates.values() for puuid in players})
    ranks = store.player_ranks(everyone)
    review.looked_up = lookup([puuid for puuid in everyone if puuid not in ranks])
    ranks.update({puuid: fields["lp_value"] for puuid, fields in review.looked_up.items()})
    for match_id, players in candidates.items():
        reason = rank_reason(players, ranks, settings)
        if reason:
            review.reasons[reason] += 1
        else:
            review.players[match_id] = players
    return review


def riot_lookup(settings: Settings):
    async def ranked(puuids: list[str]) -> dict[str, dict]:
        found = {}
        async with RiotClient(settings) as client:
            for puuid in puuids:
                try:
                    entry = pick_solo_entry(await client.league_entries(puuid))
                except NotFound:
                    continue
                if entry is not None:
                    found[puuid] = rank_fields(entry, settings.tier_floor, settings.tier_ceiling)
        return found

    def lookup(puuids: list[str]) -> dict[str, dict]:
        if not puuids or not settings.riot_api_key:
            return {}
        return asyncio.run(ranked(puuids))

    return lookup


def import_batch(settings: Settings, source, store: Store, review: Review, batch: str) -> int:
    store.open_batch(batch, SOURCE, settings.contributed_store)
    imported = []
    for match_id in review.qualified:
        if store.save_match(source.read("matches", match_id)) is False:
            continue
        store.save_timeline(match_id, source.read("timelines", match_id))
        store.tag_batch(match_id, SOURCE, batch)
        imported.append(match_id)
    for puuid in sorted({puuid for match_id in imported for puuid in review.players[match_id]} & set(review.looked_up)):
        store.upsert_player(puuid, **review.looked_up[puuid])
    store.close_batch(batch, len(imported))
    return len(imported)


def contribute(settings: Settings, confirm, source=None, lookup=None) -> dict:
    source = source or open_store(settings.contributed_store)
    with corpus_lock(settings, "a contributed import"):
        store = Store(settings)
        try:
            review = gather(settings, source, store, lookup or riot_lookup(settings))
            summary = review.summary()
            if not review.qualified:
                return {**summary, "imported": 0}
            batch = f"batch-{datetime.now():%Y%m%d-%H%M%S}"
            if not confirm(summary, batch):
                return {**summary, "imported": 0}
            return {**summary, "batch": batch, "imported": import_batch(settings, source, store, review, batch)}
        finally:
            store.close()
