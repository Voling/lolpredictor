import requests
import json
import os
import sys
import CONSTANTS
import asyncio
from pyot.models import lol
import pyot

RIOT_API_KEY = CONSTANTS.api_key

class LolModel(ModelConf):
    default_platform = "na1"
    default_region = "americas"
    default_version = "latest"
    default_locale = "en_us"

def fetchPlayer():
    file = open("test.txt", "w")

    url = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/btsnese"
    header = {"X-Riot-Token": RIOT_API_KEY}
    response = requests.get(url, headers=header)
    if response.status_code == 200:
        player = response.json()
        file.write(json.dumps(player))
        print(player["name"])
    else:
        print(response.status)
if __name__ == "__main__":
    client = pyot.Client(api_key=RIOT_API_KEY)
    fetchPlayer()