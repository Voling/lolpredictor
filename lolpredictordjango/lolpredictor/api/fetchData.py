import os
import sys

directory_path = os.path.abspath("../../lolpredictordjango")
sys.path.append(directory_path)
print(directory_path)
import pyotconf
#ignore warning, hacky way of importing using relative url
#directory path needs to be changed if folder hierarchy is changed

import asyncio
from datetime import datetime, timedelta
from pyot.models import lol
from pyot.core.queue import Queue
from pyot.utils.lol.routing import platform_to_region

RIOT_API_KEY = pyotconf.api_key
async def getMatchIds(name: str, platform: str):
    """Fetch ID's for a summoner

    Args:
        name (str): summoner name
        platform (str): platform

    Returns:
        _type_: last 100 match history ids
    """
    summoner = await lol.Summoner(name=name, platform=platform).get()
    print("PLATFORM", summoner.platform)
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

