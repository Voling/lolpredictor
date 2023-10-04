import importlib
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
async def get_match_ids(name: str, platform: str):
    summoner = await lol.Summoner(name=name, platform=platform).get()
    match_history = await lol.MatchHistory(
        puuid=summoner.puuid,
        region=platform_to_region(summoner.platform)
    ).query(
        count=100,
        queue=420,
        start_time=datetime.now() - timedelta(days=200)
    ).get()
    print(match_history.ids)
    return match_history.ids
if __name__ == "__main__":
    matchIdsCoRo = get_match_ids('btsnese','NA1')
    asyncio.run(matchIdsCoRo)