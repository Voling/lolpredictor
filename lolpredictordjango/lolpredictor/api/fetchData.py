import os
import sys

directory_path = os.path.abspath("../../lolpredictordjango")
sys.path.append(directory_path)
import pyotconf
#ignore warning, hacky way of importing using relative url
#directory path needs to be changed if folder hierarchy is changed

import asyncio
from datetime import datetime, timedelta
from pyot.models import lol
from pyot.core.queue import Queue
from pyot.utils.lol.routing import platform_to_region
from pyot.core import resources
RIOT_API_KEY = pyotconf.api_key
async def getMatchIds(name: str, platform: str):
    """Fetch ID's for a summoner

    Args:
        name (str): summoner name
        platform (str): platform

    Returns:
        _type_: last 100 match history ids
    """
    async with resources.ResourceManager():
        summoner = await lol.Summoner(name=name, platform=platform).get()
        match_history = await lol.MatchHistory(
            puuid=summoner.puuid,
            region=platform_to_region(summoner.platform)
        ).query(
            count=100,
            queue=420,
            start_time=datetime.now() - timedelta(days=200)
        ).get()
    return match_history.ids
async def getMatchFromId(id: str, platform: str):
    match = await lol.Match(
        id=id, 
        region=platform_to_region(platform.lower())).get()
    return match

async def getDetailedMatchStatistics(match_id: str, platform: str):
    """Retrieve full data of a match.

    Args:
        match_id (str):
        platform (str): na1 for starters

    Returns:
        []: participants
    """
    async with resources.ResourceManager():
        match = await lol.Match(id=match_id, region=platform).get()
        await match.get()
    return match.info.participants

async def getPlayerRank(name: str, platform: str = "na1"):
    async with resources.ResourceManager():
        summoner = await lol.Summoner(name=name, platform=platform).get()
        league_entries = await summoner.league_entries.get()
        for entry in league_entries:
            if entry.queue == "RANKED_SOLO_5x5":
                return entry.tier, entry.rank, entry.league_points
    return None

def rank_to_lp(tier: str, division: int, lp: int) -> int:
    # Define the base LP for each rank
    rank_values = {
        "IRON": 0,
        "BRONZE": 400,
        "SILVER": 800,
        "GOLD": 1200,
        "PLATINUM": 1600,
        "DIAMOND": 2000,
        "MASTER": 2400,
        "GRANDMASTER": 2600,
        "CHALLENGER": 2800
    }
    
    # Calculate the base LP for the given rank
    base_lp = rank_values.get(tier.upper(), 0)
    
    # Subtract the division value (multiplied by 100) from the base LP
    # Division 1 is higher than Division 4, so we subtract to get the correct value
    division_lp = (4 - division) * 100
    
    # Return the total LP
    return base_lp + division_lp + lp

# Example usage