from django.db import models
import datetime
# Create your models here.


class Activity(models.Model):
    activity_id = models.IntegerField()
    name = models.CharField(max_length=500)
    moving_time = models.DurationField()
    date = models.DateField(default=datetime.date.today)
    athlete_id = models.IntegerField(default=-1)
    workout_type = models.IntegerField(default=-1)

    
class Athlete(models.Model):
    athlete_id = models.IntegerField()
    first_name = models.CharField(max_length=500)
    
    
    