import os
from pyot.conf.model import activate_model, ModelConf
from pyot.conf.pipeline import activate_pipeline, PipelineConf
import sys

directory_path = os.path.abspath("./lolpredictordjango/")
sys.path.append(directory_path)
from Constants import api_key
#ignore warning, hacky way of importing using relative url
#directory path needs to be changed if folder hierarchy is changed

@activate_model("lol")
class LolModel(ModelConf):
    default_platform = "na1"
    default_region = "americas"
    default_version = "latest"
    default_locale = "en_us"


@activate_pipeline("lol")
class LolPipeline(PipelineConf):
    name = "lol_main"
    default = True
    stores = [
        {
            "backend": "pyot.stores.omnistone.Omnistone",
            "expirations": {
                "summoner_v4_by_name": 100,
                "match_v4_match": 600,
                "match_v4_timeline": 600,
            }
        },
        {
            "backend": "pyot.stores.cdragon.CDragon",
        },
        {
            "backend": "pyot.stores.riotapi.RiotAPI",
            "api_key": api_key,
        }
    ]