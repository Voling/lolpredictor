import asyncio
import logging
from typing import Any

import httpx

from ..config import Settings, get_settings
from .ratelimit import RateLimiter
from .routing import platform_host, region_for, region_host

logger = logging.getLogger(__name__)


class RiotApiError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"{status} for {url}: {body[:200]}")
        self.status = status
        self.url = url


class NotFound(RiotApiError):
    pass


class Unauthorized(RiotApiError):
    pass


class RiotClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or get_settings()
        self.region = self.settings.region or region_for(self.settings.platform)
        self._limiter = RateLimiter(
            [
                (self.settings.rate_short_requests, self.settings.rate_short_seconds),
                (self.settings.rate_long_requests, self.settings.rate_long_seconds),
            ]
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._client: httpx.AsyncClient | None = None
        self._transport = transport
        self.request_count = 0

    async def __aenter__(self) -> "RiotClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"X-Riot-Token": self.settings.riot_api_key},
            transport=self._transport,
        )
        self._semaphore = asyncio.Semaphore(self.settings.concurrency)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise RuntimeError("RiotClient must be used as an async context manager")
        assert self._semaphore is not None
        attempt = 0
        while True:
            await self._limiter.acquire()
            async with self._semaphore:
                try:
                    response = await self._client.get(url, params=params)
                except httpx.TransportError as exc:
                    if attempt >= self.settings.max_retries:
                        raise RiotApiError(0, url, str(exc)) from exc
                    attempt += 1
                    await asyncio.sleep(2**attempt)
                    continue
            self.request_count += 1
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                raise NotFound(404, url)
            if response.status_code in (401, 403):
                raise Unauthorized(response.status_code, url, response.text)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "10"))
                logger.warning("rate limited, sleeping %ss", retry_after)
                self._limiter.penalise(retry_after)
                continue
            if response.status_code >= 500 and attempt < self.settings.max_retries:
                attempt += 1
                await asyncio.sleep(2**attempt)
                continue
            raise RiotApiError(response.status_code, url, response.text)

    async def account_by_riot_id(self, name: str, tag: str) -> dict:
        url = f"{region_host(self.region)}/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
        return await self._get(url)

    async def account_by_puuid(self, puuid: str) -> dict:
        url = f"{region_host(self.region)}/riot/account/v1/accounts/by-puuid/{puuid}"
        return await self._get(url)

    async def league_entries(self, puuid: str) -> list[dict]:
        url = f"{platform_host(self.settings.platform)}/lol/league/v4/entries/by-puuid/{puuid}"
        return await self._get(url)

    async def apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> dict:
        url = f"{platform_host(self.settings.platform)}/lol/league/v4/{tier}leagues/by-queue/{queue}"
        return await self._get(url)

    async def match_ids(
        self,
        puuid: str,
        count: int = 20,
        start: int = 0,
        queue: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        url = f"{region_host(self.region)}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: dict[str, Any] = {"count": min(count, 100), "start": start}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return await self._get(url, params)

    async def match(self, match_id: str) -> dict:
        url = f"{region_host(self.region)}/lol/match/v5/matches/{match_id}"
        return await self._get(url)

    async def timeline(self, match_id: str) -> dict:
        url = f"{region_host(self.region)}/lol/match/v5/matches/{match_id}/timeline"
        return await self._get(url)
