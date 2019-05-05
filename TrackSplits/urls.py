"""TrackSplits URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from TrackSplitsApp.views import *

handler404 = 'TrackSplitsApp.views.handler404'
handler500 = 'TrackSplitsApp.views.handler500'
handler400 = 'TrackSplitsApp.views.handler400'

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', HomeView.as_view()),
    url(r'^authenticate/?', AuthenticateView.as_view()),
    url(r'^failedAuthenticate/$', AuthenticateFailedView.as_view()),
    url(r'^activities/$', ActivitiesView.as_view()),
    url(r'^activities/reload/(?P<period>[a-z]+)/$', ReloadActivitiesView.as_view()),
    url(r'^activity/(?P<id>[0-9]+)/', ActivityView.as_view()),
    url(r'^uploadActivity/', upload_file),
    url(r'^about/$', AboutView.as_view()),
    url(r'^activity_search/', get_activity_name),
    url(r'^update_splits_table', update_splits_table),
    url(r'^race_lookup', race_lookup),
        url(r'^splits_lookup', splits_lookup)
]

