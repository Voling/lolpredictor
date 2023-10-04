import sys
import os
from django.test import TestCase

directory_path = os.path.abspath("../lolpredictordjango/lolpredictor/api")
sys.path.append(directory_path)
from fetchPlayers import get_match_ids
#ignore warning, hacky way of importing using relative url


platform = 'NA1'
class MatchHistory(TestCase):
    async def testMatchHistoryIdsBtsnese(self):
        result = await get_match_ids('btsnese', platform)
        self.assertIsNotNone(result)