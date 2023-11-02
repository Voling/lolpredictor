import sys
import os
import asyncio
import json
from django.test import TestCase
from pyot.core import resources
directory_path = os.path.abspath("../lolpredictordjango/lolpredictor/api")
sys.path.append(directory_path)
from fetchData import *
#ignore warning, hacky way of importing using relative url
#directory path needs to be changed if folder hierarchy is changed

platform = 'NA1'
class MatchHistory(TestCase):
    async def testMatchHistoryIdsBtsnese(self):
        result = await getMatchIds('btsnese', platform)
        print("matchHistoryIds()[0]: {}".format(result[0]))
        self.assertIsNotNone(result)
    async def testMatchDataFromId(self):
        recentMatchId = (await getMatchIds('btsnese', platform))[0] #recent match
        result = await getMatchFromId(recentMatchId, platform)
        print("matchDataFromId(): {}".format(result.id))
        self.assertIsNotNone(result)
    async def testGetRank(self):
        result = await getPlayerRank('btsnese', platform)
        print("getRank(): {}".format(result))
        self.assertIsNotNone(result)