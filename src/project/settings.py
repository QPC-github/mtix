import os
import environ
import logging

from settings_overrider import override

logger = logging.getLogger(__name__)

# SECURITY WARNING: You must use a different value for the SECRET_KEY in
# production.  It should be generated securely, and then provided to your
# application using the DJANGO_SECRET_KEY environment variable in your
# locked-down config repository.
SECRET_KEY = '12345'

# SECURITY WARNING: DEBUG must be set to False in production! For example, set the env. var.
# in your deployment system settings: `DJANGO_DEBUG=False`.
DEBUG = True
ALLOWED_HOSTS = ['.ncbi.nlm.nih.gov']
ROOT_URLCONF = 'project.urls'
WSGI_APPLICATION = 'project.wsgi.application'

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # 3rd-party apps
    'compressor',

    # NCBI apps
    # 'django_applog',
    # 'ncbi_auth',
    # 'script_name',
    # 'ncbi.base',

    # Context passing library
    # 'ncbi_context_conduit',

    # Project apps
    'project.mtix_tmpl'
)

MIDDLEWARE = (
    'proxy_chain.middleware.ProxyChainMiddleware',
    'django.middleware.security.SecurityMiddleware',
    # 'django_applog.AppLogMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # 'ncbi_context_conduit.middleware.NCBIContextConduitMiddleware',
    # 'ncbi_auth.middleware.NCBIMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # 'django_applog.AppLogMiddleware'
)

env = environ.Env()
ENVIRONMENT = os.environ.get('DJANGO_ENVIRONMENT', 'dev')
env_fname = ".env"
if os.path.isfile(env_fname):
    logger.info(msg="Environment file: \"{}\".".format(env_fname))
    env.read_env(env_fname)
else:
    logger.warning(msg="Environment file can't be found: \"{}\".".format(env_fname))


USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')


COMPRESS_OFFLINE = True
COMPRESS_CSS_FILTERS = [
    'compressor.filters.cssmin.rCSSMinFilter',
    'compressor.filters.css_default.CssRelativeFilter'
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # 'django_applog.context_processors.generate_pinger_markup'
            ]
        }
    }
]

DATABASES = { }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'}
]

# Static files (CSS, JavaScript, Images). See
# https://docs.djangoproject.com/en/1.8/howto/static-files/ for reference.
# Note that if you are running locally with `DEBUG=False`, then you need to use
# the `--insecure` flag:
#    DJANGO_DEBUG=False manage.py runserver --insecure 0.0.0.0:9900
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join('var', 'data', 'static')
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    'compressor.finders.CompressorFinder'
)

NCBI_APP = 'mtix_tmpl'

# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'applog': {
#             'level': 'INFO',
#             'class': 'applog.AppLogHandler',
#             'appname': 'mtix_tmpl'
#         }
#     },
#     'loggers': {
#         '': {
#             'level': 'INFO',
#             'handlers': ['applog']
#         },
#         'django.request': {
#             'level': 'INFO',
#             'handlers': ['applog'],
#             'propagate': False
#         },
#         'django.security': {
#             'level': 'INFO',
#             'handlers': ['applog'],
#             'propagate': False
#         }
#     }
# }

CSRF_COOKIE_NAME = 'mtix_tmpl-csrftoken'

USE_TZ = True
TIME_ZONE = 'America/New_York'

# You can keep the private settings in ./etc/settings.yml
# You can specify path to the YAML via DJANGO_SETTINGS_YAML env variable
yaml_var = 'DJANGO_SETTINGS_YAML'
yaml_path = os.getenv(yaml_var, os.path.join('etc', 'settings.yaml'))

# You can use prefixed env variables, e.g. DJANGO_DEBUG env variable becomes DEBUG setting
var_prefix = 'DJANGO_'

if os.path.exists(yaml_path):
    override(globals(), yaml=yaml_path)

override(globals(), env=var_prefix)
