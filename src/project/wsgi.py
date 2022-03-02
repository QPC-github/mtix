import os

from django.core.wsgi import get_wsgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")

_application = get_wsgi_application()


def application(environ, start_response):
    script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
    if script_name:
        environ['SCRIPT_NAME'] = script_name

    return _application(environ, start_response)
