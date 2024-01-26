import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..ranks import pick_solo_entry, rank_to_lp, tier_index
from ..riot.client import NotFound, RiotApiError, RiotClient, Unauthorized
from ..riot.routing import split_riot_id
from .store import Store

logger = logging.getLogger(__name__)

MIN_DURATION_SECONDS = 600


@dataclass
class CrawlReport:
    matches_added: int = 0
    timelines_added: int = 0
    players_crawled: int = 0
    players_discovered: int = 0
    players_out_of_scope: int = 0
    players_identified: int = 0
    requests: int = 0
    budget: int = 0
    errors: list[str] = field(default_factory=list)
    leaderboard: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "matches_added": self.matches_added,
            "timelines_added": self.timelines_added,
            "players_crawled": self.players_crawled,
            "players_discovered": self.players_discovered,
            "players_out_of_scope": self.players_out_of_scope,
            "players_identified": self.players_identified,
            "requests": self.requests,
            "budget": self.budget,
            "errors": self.errors[:20],
            "leaderboard": self.leaderboard,
        }


class Seeder:
    def __init__(self, store: Store, client: RiotClient, settings: Settings | None = None):
        self.store = store
        self.client = client
        self.settings = settings or get_settings()
        self.report = CrawlReport()

    async def seed(self, riot_id: str | None = None) -> str:
        riot_id = riot_id or self.settings.seed_riot_id
        name, tag = split_riot_id(riot_id)
        cached = self.store.find_player_by_riot_id(name, tag)
        if cached:
            puuid = cached["puuid"]
        else:
            account = await self.client.account_by_riot_id(name, tag)
            puuid = account["puuid"]
            self.store.upsert_player(
                puuid, game_name=account.get("gameName", name), tag_line=account.get("tagLine", tag), depth=0
            )
        self.store.push_frontier(puuid, depth=0, priority=1e9, requeue=True)
        return puuid

    async def seed_leaderboard(self) -> dict:
        tiers = [tier.strip().lower() for tier in self.settings.apex_tiers.split(",") if tier.strip()]
        floor = self.settings.apex_min_league_points
        seeded, distribution, skipped = 0, {}, 0
        for tier in tiers:
            try:
                league = await self.client.apex_league(tier)
            except Unauthorized:
                raise
            except (NotFound, RiotApiError) as exc:
                self.report.errors.append(f"leaderboard {tier}: {exc}")
                continue
            entries = league.get("entries") or []
            points = sorted(int(entry.get("leaguePoints", 0)) for entry in entries)
            distribution[tier] = {
                "players": len(entries),
                "above_floor": sum(1 for value in points if value >= floor),
                "median_lp": points[len(points) // 2] if points else 0,
                "max_lp": points[-1] if points else 0,
            }
            for entry in entries:
                puuid = entry.get("puuid")
                league_points = int(entry.get("leaguePoints", 0))
                if not puuid or league_points < floor:
                    skipped += 1
                    continue
                self.store.upsert_player(
                    puuid,
                    tier=tier.upper(),
                    league_points=league_points,
                    lp_value=rank_to_lp(tier.upper(), "I", league_points),
                    wins=entry.get("wins"),
                    losses=entry.get("losses"),
                    in_scope=True,
                    depth=0,
                )
                self.store.push_frontier(puuid, depth=0, priority=float(league_points), requeue=True)
                seeded += 1
        return {"seeded": seeded, "below_floor": skipped, "floor": floor, "distribution": distribution}

    def _spent(self) -> int:
        return self.client.request_count

    def _crawl_budget_left(self) -> int:
        ceiling = self.settings.max_requests - self.settings.identify_reserve
        return ceiling - self._spent()

    def _budget_left(self) -> int:
        return self.settings.max_requests - self._spent()

    def _in_scope(self, tier: str | None) -> bool:
        index = tier_index(tier)
        if index < 0:
            return False
        return self.settings.tier_floor <= index <= self.settings.tier_ceiling

    async def _refresh_rank(self, puuid: str) -> dict | None:
        try:
            entries = await self.client.league_entries(puuid)
        except NotFound:
            return None
        entry = pick_solo_entry(entries)
        if entry is None:
            self.store.upsert_player(puuid, tier=None, in_scope=False)
            return None
        fields = {
            "tier": entry.get("tier"),
            "division": entry.get("rank"),
            "league_points": entry.get("leaguePoints"),
            "lp_value": rank_to_lp(entry.get("tier"), entry.get("rank"), entry.get("leaguePoints")),
            "wins": entry.get("wins"),
            "losses": entry.get("losses"),
            "in_scope": self._in_scope(entry.get("tier")),
        }
        self.store.upsert_player(puuid, **fields)
        return fields

    async def _identify(self, puuid: str) -> None:
        player = self.store.get_player(puuid)
        if player and player.get("game_name"):
            return
        try:
            account = await self.client.account_by_puuid(puuid)
        except (NotFound, Unauthorized):
            return
        self.store.upsert_player(
            puuid, game_name=account.get("gameName"), tag_line=account.get("tagLine")
        )

    async def _ingest_match(self, match_id: str, depth: int) -> None:
        if not self.store.has_match(match_id):
            match = await self.client.match(match_id)
            info = match.get("info", {})
            if info.get("queueId") != self.settings.queue_id:
                return
            if int(info.get("gameDuration", 0)) < MIN_DURATION_SECONDS:
                return
            self.store.save_match(match)
            self.report.matches_added += 1
        else:
            match = self.store.load_match(match_id)
        if self.settings.fetch_timelines and not self.store.has_timeline(match_id):
            try:
                timeline = await self.client.timeline(match_id)
                self.store.save_timeline(match_id, timeline)
                self.report.timelines_added += 1
            except NotFound:
                pass
        for participant in match["info"]["participants"]:
            puuid = participant["puuid"]
            known = self.store.get_player(puuid)
            if known is None:
                self.store.upsert_player(puuid, depth=depth + 1)
                self.report.players_discovered += 1
            if depth + 1 <= self.settings.max_depth:
                self.store.push_frontier(puuid, depth + 1, priority=1.0)

    def _since(self) -> int | None:
        days = self.settings.crawl_since_days
        if days <= 0:
            return None
        return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    async def _match_ids(self, puuid: str) -> list[str]:
        wanted = self.settings.matches_per_player
        collected: list[str] = []
        since = self._since()
        while len(collected) < wanted and self._crawl_budget_left() > 2:
            page = await self.client.match_ids(
                puuid,
                count=min(100, wanted - len(collected)),
                start=len(collected),
                queue=self.settings.queue_id,
                start_time=since,
            )
            collected.extend(page)
            if len(page) < 100:
                break
        return collected

    async def _crawl_player(self, puuid: str, depth: int) -> None:
        known = self.store.get_player(puuid) if depth == 0 else None
        if known and known.get("tier") and known.get("in_scope"):
            rank = {"in_scope": True}
        else:
            rank = await self._refresh_rank(puuid)
        if depth > 0 and (rank is None or not rank.get("in_scope")):
            self.store.mark_frontier(puuid, "out_of_scope")
            self.report.players_out_of_scope += 1
            return
        await self._identify(puuid)
        try:
            ids = await self._match_ids(puuid)
        except NotFound:
            self.store.mark_frontier(puuid, "error")
            return
        pending = [match_id for match_id in ids if not self.store.has_match(match_id)]
        pending += [match_id for match_id in ids if match_id not in pending]
        width = max(self.settings.concurrency, 1)
        for start in range(0, len(pending), width):
            if self.report.matches_added >= self.settings.max_matches:
                break
            if self._crawl_budget_left() <= 2 * width:
                break
            chunk = pending[start : start + width]
            results = await asyncio.gather(
                *(self._ingest_match(match_id, depth) for match_id in chunk), return_exceptions=True
            )
            for match_id, outcome in zip(chunk, results):
                if isinstance(outcome, Unauthorized):
                    raise outcome
                if isinstance(outcome, BaseException):
                    self.report.errors.append(f"{match_id}: {outcome}")
        self.store.mark_frontier(puuid, "done")
        self.store.upsert_player(puuid, crawled_at=datetime.now(timezone.utc))
        self.report.players_crawled += 1

    async def run(self, riot_id: str | None = None, leaderboard: bool = False) -> CrawlReport:
        self.store.reset_active()
        if leaderboard:
            self.report.leaderboard = await self.seed_leaderboard()
        else:
            await self.seed(riot_id)
        while True:
            if self.report.matches_added >= self.settings.max_matches:
                break
            if self.report.players_crawled >= self.settings.max_players:
                break
            if self._crawl_budget_left() <= 4:
                logger.info("crawl budget spent, stopping at %s requests", self._spent())
                break
            batch = self.store.pop_frontier(1)
            if not batch:
                break
            entry = batch[0]
            try:
                await self._crawl_player(entry["puuid"], int(entry["depth"] or 0))
            except Unauthorized as exc:
                self.report.errors.append(str(exc))
                break
            except Exception as exc:
                logger.warning("crawl failed for %s: %s", entry["puuid"], exc, exc_info=True)
                self.report.errors.append(f"{entry['puuid']}: {exc}")
                self.store.mark_frontier(entry["puuid"], "error")
            logger.info(
                "crawled=%s matches=%s timelines=%s requests=%s",
                self.report.players_crawled,
                self.report.matches_added,
                self.report.timelines_added,
                self.client.request_count,
            )
        await self._identify_pass()
        self.report.requests = self.client.request_count
        self.report.budget = self.settings.max_requests
        return self.report

    async def _identify_pass(self) -> None:
        self.report.players_identified += self.store.backfill_identities()
        for puuid in self.store.unnamed_players(self.settings.min_profile_games):
            if self._budget_left() <= 1:
                break
            try:
                await self._identify(puuid)
            except Unauthorized:
                break
            except Exception as exc:
                self.report.errors.append(f"identify {puuid}: {exc}")
                continue
            self.report.players_identified += 1


async def refresh_ranks(settings: Settings | None = None, min_games: int = 1) -> dict:
    settings = settings or get_settings()
    checked, ranked, failed = 0, 0, 0
    with Store(settings) as store:
        async with RiotClient(settings) as client:
            seeder = Seeder(store, client, settings)
            for puuid in store.unranked_players(min_games):
                if client.request_count >= settings.max_requests:
                    break
                try:
                    fields = await seeder._refresh_rank(puuid)
                except Unauthorized:
                    break
                except Exception:
                    failed += 1
                    continue
                checked += 1
                if fields:
                    ranked += 1
    return {"checked": checked, "ranked": ranked, "failed": failed, "requests": checked + failed}


async def crawl(
    riot_id: str | None = None, settings: Settings | None = None, leaderboard: bool = False
) -> CrawlReport:
    settings = settings or get_settings()
    with Store(settings) as store:
        async with RiotClient(settings) as client:
            return await Seeder(store, client, settings).run(riot_id, leaderboard=leaderboard)
