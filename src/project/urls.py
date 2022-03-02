from django.conf.urls import include
from django.contrib import admin
from django.urls import path

import ncbi.base.views
import project.mtix_tmpl.urls


urlpatterns = [
    path('', include(project.mtix_tmpl.urls, namespace='project-mtix_tmpl')),
    path(r'admin/', admin.site.urls)
]


handler400 = ncbi.base.views.ncbi_bad_request
handler403 = ncbi.base.views.ncbi_permission_denied
handler404 = ncbi.base.views.ncbi_page_not_found
handler500 = ncbi.base.views.ncbi_server_error
