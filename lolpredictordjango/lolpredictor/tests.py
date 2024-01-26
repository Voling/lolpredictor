import sys
import os
directory_path = os.path.abspath("../lolpredictordjango/lolpredictor/api")
sys.path.append(directory_path)
directory_path = os.path.abspath("../lolpredictordjango/lolpredictordjango/")
sys.path.append(directory_path)

import asyncio
import json
import Constants
from django.test import TestCase

from fetchData import *
platform = 'americas'
class MatchHistory(TestCase):
    async def testMatchHistoryIdsBtsnese(self):
        result = await getLastMatches('vawowant', 'sage', 'na1', platform, 20)
        print("matchHistoryIds()[0]: {}".format(result[1]))
        self.assertIsNotNone(result)