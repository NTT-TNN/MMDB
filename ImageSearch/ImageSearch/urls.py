
from django.conf.urls import url,include
from django.contrib import admin
from searchApp import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^admin/', admin.site.urls),
    url(r'^searchApp/', include('searchApp.urls')),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    url(r'^photos/', include('photos.urls', namespace='photos')),
]
if settings.DEBUG is True:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
