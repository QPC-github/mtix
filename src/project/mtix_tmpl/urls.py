from django.urls import path

from project.mtix_tmpl.views import cvHealthcheck, cStopKafka, cKafkaEnv
from django.conf.urls import url
from django.contrib import admin

app_name = "project.mtix_tmpl"

# NOTE: the value of name in your UrlConf will be written to AppLog as ncbi_pdid (page design ID)
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^health', cvHealthcheck.as_view(), name='health'),
    url(r'^stop', cStopKafka.as_view()),
    url(r'^env', cKafkaEnv.as_view()),
    path('', cKafkaEnv.as_view())
]
