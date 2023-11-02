from django.contrib import admin

#Register my models from models.py here
from .models import Summoner, Match, MatchHistory, MatchHistoryData, Rank
#Register all of the models
admin.site.register(Summoner)
admin.site.register(Match)
admin.site.register(MatchHistory)
admin.site.register(MatchHistoryData)
admin.site.register(Rank)