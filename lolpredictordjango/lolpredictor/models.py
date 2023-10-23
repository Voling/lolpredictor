from django.db import models

class Player(models.Model):
    #for general statistics only
    name = models.CharField(max_length=10)
    winrate = models.FloatField
    def saveData(self, data):
        self.name = data['name']
        self.save()
class ParticipationInMatch(models.Model):
    match_id = models.CharField(max_length=16)
    kills = models.IntegerField
    deaths = models.IntegerField
    damage_dealt = models.IntegerField
    damage_taken = models.IntegerField
    

