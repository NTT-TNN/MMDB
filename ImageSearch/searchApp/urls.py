# from django.urls import path
from django.conf.urls import url
from searchApp import views
urlpatterns = [
    # path('', views.index, name='index'),
    url(r'search', views.search, name='search'),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
]