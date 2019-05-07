# Generated by Django 2.0.13 on 2019-05-07 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('TrackSplitsApp', '0004_activity_workout_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='activity_id',
            field=models.BigIntegerField(),
        ),
        migrations.AlterField(
            model_name='activity',
            name='athlete_id',
            field=models.BigIntegerField(default=-1),
        ),
        migrations.AlterField(
            model_name='athlete',
            name='athlete_id',
            field=models.BigIntegerField(),
        ),
    ]