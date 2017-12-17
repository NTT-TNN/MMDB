
from django.conf.urls import url,include
from django.contrib import admin
from searchApp import views
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^admin/', admin.site.urls),
    url(r'^searchApp/', include('searchApp.urls')),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
]
