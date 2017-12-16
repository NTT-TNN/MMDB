
from django.conf.urls import url
from django.contrib import admin
from searchApp import views
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^admin/', admin.site.urls),
    url(r'search/(?P<key>\d+)', views.search, name='search'),
]
