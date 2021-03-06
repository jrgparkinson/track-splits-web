# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-12-26 21:06
from __future__ import unicode_literals

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('TrackSplitsApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Athlete',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('athlete_id', models.IntegerField()),
                ('first_name', models.CharField(max_length=500)),
            ],
        ),
        migrations.AddField(
            model_name='activity',
            name='date',
            field=models.DateField(default=datetime.date.today),
        ),
    ]
