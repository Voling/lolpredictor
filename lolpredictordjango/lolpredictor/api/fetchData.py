import os
import sys

import httpx
import asyncio

import Constants
RIOT_API_KEY = Constants.api_key
async def getLastMatches(name, tagline, region, platform, amount):
    base_url = f"https://{platform}.api.riotgames.com/lol"
    summoner_url = f"{base_url}/summoner/v1/summoners/by-riot-id/{name}/{tagline}"
    headers = {"X-Riot-Token": RIOT_API_KEY}

    async with httpx.AsyncClient() as client:
        response = await client.get(summoner_url, headers=headers)
        print(response.content)
        if response.status_code != 200:
            return None, f"Error fetching summoner data: {response.status_code}"

        summoner_id = response.json()['id']
        matches_url = f"{Constants.base_url}/match/v4/matchlists/by-account/{summoner_id}?endIndex={amount}"
        response = await client.get(matches_url, headers=headers)
        if response.status_code != 200:
            return None, f"Error fetching match data: {response.status_code}"

        return response.json()