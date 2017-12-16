from django.urls import path
from django.conf.urls import url
from . import views
from searchApp import views
urlpatterns = [
    path('', views.index, name='index'),
    url(r'search', views.search, name='search'),
]