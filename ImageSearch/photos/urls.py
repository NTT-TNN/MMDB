from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'^clear/$', views.clear_database, name='clear_database'),
    url(r'^$', views.BasicUploadView.as_view(), name='basic_upload'),
]
