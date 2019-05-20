from __future__ import unicode_literals
from django.shortcuts import render
from django.views.generic import TemplateView, RedirectView
from django.contrib.sites.shortcuts import get_current_site
from django.http import HttpResponse
from stravalib.client import Client
import stravalib
from django.conf import settings
from django.shortcuts import redirect
from django.template.loader import render_to_string
import numpy as np
import django_tables2 as tables
import sys
from .TrackFit import TrackFit as TF
from TrackSplitsApp.models import Activity, Athlete
import json
import datetime as DT
from .forms import UploadFileForm
import gpxpy
import gpxpy.gpx


def handle_uploaded_file(f):
    print('Handle uploaded file')

    file_contents_bytes = f.read()

    # Need to convert from bytes to a string
    file_contents = file_contents_bytes.decode()

    print('Finished reading file')

    # file_contents = file_contents.replace("\\n", "  ")

    print('Finished replacing')

    # print(file_contents)

    gpx = gpxpy.parse(file_contents)

    print(gpx)

    # TODO: get latlng and times
    latlng = []
    times = []
    total_distance = float('NaN')

    success = False
    activity_name = ''

    # For now, just analyse first track/segment

    if gpx.tracks:

        track = gpx.tracks[0]

        activity_name= track.name

        if track.segments:

            segment = track.segments[0]

            print(segment)

            total_distance = segment.length_2d()

            for pt in segment.points:

                latlng.append([pt.latitude, pt.longitude])
                times.append(pt.time)

            if len(latlng) > 0:
                success = True

            # Convert formats:
            latlng = np.array(latlng)

            # Convert time from timedelta objects to time in seconds since start of activity
            times = [(t-times[0]).total_seconds() for t in times]

            print('Processed GPX segment, distance = %g metres' % total_distance)
            # print('Latlng: ' + str(latlng))

    activity = {'name': activity_name, 'success': success,
                'latlng': latlng, 'times': times, 'total_distance': total_distance}

    return activity

def upload_file(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            activity = handle_uploaded_file(request.FILES['file'])
            # return render(request, 'analyseFile.html', )

            if not activity['success']:
                return render(request, 'upload.html', {'form': form, 'upload_error': True})


            gps_distance = activity['total_distance']
            race = guess_race(gps_distance)

            # races = get_races()
            context = {}
            # guessed_race_distance = 5000


            split_distances = generate_split_distances_race(race)

            # race = races[guessed_race_distance]
            stravaSessionExists = False
            isStrava = False

            # Get latlng, times from GPX
            latlng = activity['latlng']
            times = activity['times']


            track_length = race['lap_length']
            laps = []
            lap_index = -1
            [splits, trackFit] = get_splits(latlng, times, race['distance'], track_length, laps, lap_index,
                                            split_distances, verbosity=1)

            context = make_activity_context(context,
                          splits, activity,
                          race,
                          stravaSessionExists,
                          isStrava,
                          trackFit, laps, lap_index)

            request.session['laps'] = json.dumps(laps)
            request.session['latlng'] = trackFit.latlng.tolist()  # latlng.tolist()
            request.session['times'] = trackFit.times  # times


            return render(request, 'activity.html', context)
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form, 'upload_error': False})


class HomeView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):

        print('HomeView')

        client = Client()

        context = super(HomeView, self).get_context_data(**kwargs)

        context['stravaSessionExists'] = False

        #full_url = 'http://localhost:5000/' # TODO - make this work for live deployment too
        #full_url = full_url + 'authenticate/'
        full_url = self.request.build_absolute_uri('authenticate/')
        authorize_url = client.authorization_url(client_id=settings.STRAVA_CLIENT_ID, redirect_uri=full_url)

        context['authenticateLink'] = authorize_url

        if user_connected(self.request):

            client.access_token = self.request.session['access_token']
            
            try:
                athlete = client.get_athlete()
                context['stravaSessionExists'] = True
                
            except stravalib.exc.AccessUnauthorized:
                print('Exception - stravalib.exc.AccessUnauthorized')
                
            except:
                print('An error occured getting the athlete which we do not recognise')
                
                
            if context['stravaSessionExists']:
                context['athlete'] = athlete

                context['firstTime'] = False
                try:
                    context['firstTime'] = self.request.session['first_time']
                except:
                    pass
                self.request.session['first_time'] = False

            # Remove this eventually!!
            # Just remove my (Jamie's) activities
            # if athlete.id == 1619378:
            #     Activity.objects.filter(athlete_id=athlete.id).delete()

        return context


class AboutView(TemplateView):
    template_name = 'about.html'

    print('About view')

    def get_context_data(self, **kwargs):
        context = super(AboutView, self).get_context_data(**kwargs)
        if user_connected(self.request):
            context['stravaSessionExists'] = True

        return context


def update_splits_table(request):
    try:
        race = request.GET.get('race')
        race_dist = float(race.replace('SC', ''))
        lap_length = float(request.GET.get('lap_length'))
        lap_index = 0
        try:
            lap_index = int(request.GET.get('lap_index'))
        except:
            pass

        # I doubt this will work
        req_splits = request.GET.get('splits')
        req_splits = req_splits.split(',')
        req_splits = [float(x) for x in req_splits]
        #print('request splits:' + str(req_splits))
        split_distances = np.array(req_splits)

        # Need to somehow retrieve latlng and times
        latlng = np.array(request.session['latlng'])
        times = request.session['times']

        laps = json.loads(request.session['laps'])

        #print('Recovered latlng: ' + str(latlng))
        #print('Recovered times: ' + str(times))

        verbosity = 1
        [splits, trackFit] = get_splits(latlng, times, race_dist, lap_length, laps, lap_index, split_distances,
                                        verbosity)

        distance_time = get_distance_time(trackFit)

        activity_latlng = trackFit.latlng
        fit_latlng = trackFit.fit_latlng
        activity_centre = trackFit.get_centre(trackFit.latlng)

        fitError = get_fit_error(trackFit, splits)

        angle_bins_freqs = trackFit.bins_freqs
        accepted_angles_limits = trackFit.accepted_angles_limits

    except:

        splits = []
        distance_time = []
        activity_latlng = []
        fit_latlng = []
        activity_centre = [0, 0]

        fitError = {}

        angle_bins_freqs = []
        accepted_angles_limits = [0, 0]

    html = render_to_string('activity-reloadable.html', {'splits': splits, 'distance_time': distance_time,
                                                         'activity_latlng': activity_latlng, 'fit_latlng': fit_latlng,
                                                         'activity_centre': activity_centre, 'fitError': fitError,
                                                         'accepted_angles_limits': accepted_angles_limits,
                                                         'angle_bins_freqs': angle_bins_freqs})


    #json_data = json.dumps({'splits_table': splits_table_html, 'distance-plot': distance_plot_html})
    #json_data = json.dumps(combined_html)
    #return HttpResponse(json_data,content_type='application/json')
    return HttpResponse(html)


def get_distance_time(trackFit):
    distance_time = []
    for i in range(0, len(trackFit.D)):
        # Making sure convert times to milliseconds
        if i == 0:
            speed = (trackFit.D[i + 1] - trackFit.D[i]) / (trackFit.times[i + 1] - trackFit.times[i])
        elif i == len(trackFit.D) - 1:
            speed = (trackFit.D[i] - trackFit.D[i - 1]) / (trackFit.times[i] - trackFit.times[i - 1])
        else:
            speed = (trackFit.D[i + 1] - trackFit.D[i - 1]) / (trackFit.times[i + 1] - trackFit.times[i - 1])

        dist_time = {'distance': trackFit.D[i], 'time': float(trackFit.times[i]) * 1000, 'speed': speed}
        distance_time.append(dist_time)

    return distance_time


def splits_lookup(request):
    interval = float(request.GET.get('split_interval'))
    race = request.GET.get('race')

    distance = float(str(race).replace('SC', ''))

    new_splits = generate_split_distances(distance, interval)
    new_splits_str = [str(int(split)) for split in new_splits]

    data = {}
    data['splits'] = ','.join(new_splits_str)

    return HttpResponse(json.dumps(data))


def race_lookup(request):
    requested_race = request.GET.get('race')

    race = None

    for race in ActivityView.RACES:
        if race['display_name'] == requested_race:
            break

    #race['start_location'] = race['lap_length'] - race['distance'] % race['lap_length']

    split_distances = generate_split_distances_race(race)
    split_strs = [str(split) for split in split_distances]
    race['splits'] = ','.join(split_strs)

    return HttpResponse(json.dumps(race))


def get_splits(latlng, times, guessed_race_distance, track_length, laps, lap_index, split_distances, verbosity = 0):
    trackFit = TF(latlng, times, guessed_race_distance, track_length, laps, lap_index, verbosity)

    splits = trackFit.compute_splits(split_distances)

    return [splits, trackFit]

def one_week_ago():
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=7)
    since = week_ago.isoformat()
    
    return since

class ReloadActivitiesView(RedirectView):
    def get(self, request, *args, **kwargs):
        if not user_connected(request):
            self.url = '/'
            return super(ReloadActivitiesView, self).get(request, *args, **kwargs)

        client = Client()
        client.access_token = self.request.session['access_token']

        period = self.kwargs['period']
        since = None

        if period == 'week':
            since = one_week_ago()

        # This gets and saves all activities
        get_all_activities(client, self.request.session['athlete_id'], since)

        self.url = '/activities/'
        return super(ReloadActivitiesView, self).get(request, *args, **kwargs)


class AuthenticateView(RedirectView):
    # This should just authenticate, start a session and then redirect to the activities viewitems

    def get(self, request, *args, **kwargs):
        try:
            #print(request.GET)
            code = request.GET.get('code')
        except:
            code = None

        print('Code: ' + str(code))

        client = Client()
        client.access_token = None
        if code:
            access_token = client.exchange_code_for_token(client_id=settings.STRAVA_CLIENT_ID,
                                                                 client_secret=settings.STRAVA_SECRET, code=code)
            
            print('Got access token: %s' % client.access_token)
            
            # Just set the actual token for the client?
            actual_token = access_token['access_token']
            client.access_token = actual_token
            
            print('Set client.access_token = %s' % client.access_token)

            athlete = client.get_athlete()


        # Now store that access token somewhere (a database?)
        if client.access_token:
            request.session['access_token'] = client.access_token
            request.session['athlete_id'] = athlete.id
            request.session['first_time'] = True
            self.url = '/activities/'
            return super(AuthenticateView, self).get(request, *args, **kwargs)

        else:
            request.session['access_token'] = None
            self.url = '/failedAuthenticate/'
            return super(AuthenticateView, self).get(request, *args, **kwargs)



            # If authentication failed, display some message here:


class AuthenticateFailedView(TemplateView):
    # This should probably just be some subset of the home template
    template_name = 'authenticate.html'


# Since should be a date like 2017-12-20
def get_all_activities(client, athlete_id, since=None):
    print('Getting activities from strava since ' + str(since))


    if since:
        stravaActivities = client.get_activities(after=since + "T00:00:00Z")
    else:
        stravaActivities = client.get_activities()

        print('Got activities from strava')

    for act in stravaActivities:
        
        if not act.workout_type:
            workoutType = 0
        else:
            workoutType = int(act.workout_type)
            
        print('Adding activity: ' + act.name + ", " + str(act.moving_time))
            
        modelActivity, created = Activity.objects.get_or_create(activity_id=act.id, name=act.name,
                                                                moving_time=act.moving_time, date=act.start_date,
                                                                athlete_id=athlete_id, workout_type=workoutType)

        if created:
            print('Done')
        else:

            print('Already had activity')

        modelActivity.save()


class ActivitiesView(TemplateView):
    template_name = 'activities.html'


    def get_context_data(self, **kwargs):
        #self.dispatch(self.request)
        if 'access_token' not in self.request.session:
            self.request.session['access_token'] = None
            print('No access_token')
            return
        
        print('Activities view')

        client = Client()
        client.access_token = self.request.session['access_token']
        athlete = client.get_athlete()
        print("For athlete id: {id}, I now have an access token: {token}".format(id=athlete.id, token=client.access_token))

        context = super(ActivitiesView, self).get_context_data(**kwargs)

        # First try and get all activities for this athlete from the DB, to avoid hitting strava
        activities = Activity.objects.all().filter(athlete_id=self.request.session['athlete_id'])
        print('Activities found: ' + str(activities))

        if len(activities) == 0:
            print('No activities for athlete, querying Strava')
            get_all_activities(client, self.request.session['athlete_id'])
            activities = Activity.objects.all().filter(athlete_id=self.request.session['athlete_id'])

        activities = activities.order_by('-date')
        context['activities'] = activities
        context['stravaSessionExists'] = True

        return context


    def dispatch(self, request, *args, **kwargs):
        req = self.request

        if not user_connected(request):
            return redirect('/')

        return super(ActivitiesView, self).dispatch(request, *args, **kwargs)


def get_activity_name(request):
    print('get_activity_name')
    if request.is_ajax():
        q = request.GET.get('term', '')
        activities = Activity.objects.filter(name__icontains=q)[:20]
        #print(activities)
        results = []
        for activity in activities:
            activity_json = {}
            activity_json['id'] = activity.activity_id
            activity_json['label'] = activity.name
            activity_json['value'] = activity.name
            results.append(activity_json)

        print(results)
        data = json.dumps(results)
    else:
        data = 'fail'
    mimetype = 'application/json'
    return HttpResponse(data, mimetype)

def guess_race(gps_distance):
    """ Given the distance of an activity, guess what race it was """
    race = None
    races = get_races()

    race_dists = [dist['distance'] for dist in races]

    guessed_race_distance = min(race_dists, key=lambda x: abs(x - gps_distance))

    # First try searching non S/C
    for r in races:
        if r['distance'] == guessed_race_distance and 'SC' not in r['display_name']:
            race = r

    # If we didn't find a race, allow S/C
    if not race:
        for r in races:
            if r['distance'] == guessed_race_distance:
                race = r

    return race

def get_races():
    races = [{'distance': 1500, 'display_name': '1500', 'lap_length': 400},
             {'distance': 1609, 'display_name': 'Mile', 'lap_length': 400},
             {'distance': 3000, 'display_name': '3000', 'lap_length': 400},
             {'distance': 5000, 'display_name': '5000', 'lap_length': 400},
             {'distance': 10000, 'display_name': '10000', 'lap_length': 400},
             {'distance': 3000, 'display_name': '3000SC', 'lap_length': 390},
             {'distance': 2000, 'display_name': '2000SC', 'lap_length': 390},
             {'distance': 1500, 'display_name': '1500SC', 'lap_length': 390}
    ]

    return races

class ActivityView(TemplateView):
    template_name = 'activity.html'

    RACES = get_races()

    def get_context_data(self, **kwargs):
        context = super(ActivityView, self).get_context_data(**kwargs)

        activity_id = self.kwargs['id']

        # Get the activity for this id
        client = Client()
        client.access_token = self.request.session['access_token']
        athlete = client.get_athlete()
        print("For athlete id: {id}, I now have an access token: {token}".format(id=athlete.id,
                                                                                 token=client.access_token))

        types = ['time', 'latlng']
        streams = client.get_activity_streams(activity_id, types=types, resolution='medium')
        latlng = np.array(streams['latlng'].data)  # NOTE: Latitude is first index, longitude second
        times = streams['time'].data
        strava_activity = client.get_activity(activity_id)

        # Also get the laps
        lapsIterator = client.get_activity_laps(activity_id)
        laps = []
        for lap in lapsIterator:
            # Choose which lap properties we'll keep here
            formatted_time = str(lap.elapsed_time)
            laps.append({'lap_index': lap.lap_index, 'elapsed_time': lap.elapsed_time.total_seconds(),
                         'start_index': lap.start_index, 'name': lap.name, 'formatted_time': formatted_time,
                         'distance': float(lap.distance)})





        # context['latlng'] = latlng

        lap_index = 0
        num_laps = len(laps)

        race = None

        # Search all laps to try and get a fit
        # lap_index of -1 means ignore laps and consider entire activity
        for lap_index in range(0, num_laps + 1):

            # First need to work out what the distance of the race is
            # Round to nearest 100m
            if lap_index == 0:
                strava_dist = int(strava_activity.distance)
            else:
                strava_dist = int(laps[lap_index - 1]['distance'])

            race = guess_race(strava_dist)

            track_length = race['lap_length']

            trackFit = TF(latlng, times, race['distance'], track_length, laps, lap_index, 1)

            # errorTolerance = 15.0  # metres
            # if not (trackFit.fit_error < errorTolerance):
            #     context['successfulFit'] = False
            # else:
            #     context['successfulFit'] = True
            #     break

            if trackFit.is_good_fit():
                break

        # Use 400m splits for 2000m races or less, 1k for longer


        split_distances = generate_split_distances_race(race)
        # splits = trackFit.compute_splits(split_distances)

        [splits, trackFit] = get_splits(latlng, times, race['distance'], race['lap_length'], laps, lap_index,
                                        split_distances, 1)

        # Define variables needed for making html page
        stravaSessionExists = True
        isStrava = True
        context = make_activity_context(context,
                                        splits, strava_activity,
                                        race,
                                        stravaSessionExists,
                                        isStrava,
                                        trackFit, laps, lap_index)

        # Store these so that if we wish to compute different splits we don't have to hit Strava servers again
        self.request.session['laps'] = json.dumps(laps)
        self.request.session['latlng'] = trackFit.latlng.tolist() # latlng.tolist()
        self.request.session['times'] = trackFit.times #times

        return context


    def dispatch(self, request, *args, **kwargs):
        req = self.request

        if not user_connected(request):
            return redirect('/')

        return super(ActivityView, self).dispatch(request, *args, **kwargs)


def get_fit_error(trackFit, splits):
    speed = float(splits[-1]['distance']) / float(splits[-1]['time'])

    err = {'distance': round(0.5 * trackFit.fit_error, 1), 'time': round(0.5 * trackFit.fit_error / speed, 1),
           'speed': round(speed, 1), 'speedMPH': round(speed * 2.24, 1)}

    return err

def make_activity_context(context,
                          splits, strava_activity,
                          race,
                          stravaSessionExists,
                          isStrava,
                          trackFit,
laps,                          lap_index=-1
                          ):

    """
    A
    :param context: dict
    :param guessed_race_distance: float
    :param splits: List[float]
    :param strava_activity: StravaActivity object, needs at least a 'name' field, and an 'id' field if Strava
    :param race: dict
    :param stravaSessionExists: bool
    :param isStrava:  bool
    :param trackFit: TrackFit object
    :param lap_index: lap index to consider (int). Set to -1 to consider entire activity.
    :param laps: ??
    :return:
    """

    context['lap_index'] = lap_index
    context['laps'] = laps

    distance_time = get_distance_time(trackFit)
    track_length = race['lap_length']

    context['race_dist'] = race['distance']
    context['track_length'] = track_length
    context['splits'] = splits
    context['activity'] = strava_activity
    context['RACES'] = get_races()
    context['race'] = race
    context['stravaSessionExists'] = stravaSessionExists
    context['isStrava'] = isStrava
    context['activity_latlng'] = trackFit.latlng
    context['fit_latlng'] = trackFit.fit_latlng
    context['activity_centre'] = trackFit.get_centre(trackFit.latlng)
    context['distance_time'] = distance_time
    context['angle_bins_freqs'] = trackFit.bins_freqs
    context['accepted_angles_limits'] = trackFit.accepted_angles_limits
    context['fitError'] = get_fit_error(trackFit, splits)
    context['successfulFit'] = trackFit.is_good_fit()
    context['latlng'] = trackFit.latlng

    return context

def generate_split_distances_race(race):
    #track_length = race['lap_length']

    distance = race['distance']
    interval = race['lap_length']
    
    if distance > 2000:
        interval = 1000
    
    return generate_split_distances(distance, interval)
    
def generate_split_distances(distance, interval):
    
    i = 1
    split_distances = []
    while i * interval <= distance:
        split_distances.append(i * interval)
        i = i + 1
        print(split_distances)

    if split_distances[-1] < distance:
        split_distances.append(distance)

    return split_distances


def user_connected(request):
    if not 'access_token' in request.session or not request.session['access_token']:
        return False

    return True


def handler404(request, exception):
    data = {}
    return render(request, '404.html', data)


def handler500(request):
    data = {}
    return render(request, '500.html', data)


def handler400(request, exception):
    data = {}
    return render(request, '400.html', data)
