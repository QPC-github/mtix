from project.mtix_tmpl.kafka_utils import get_env_variable


class cAppConfig():
    def __init__(self):
        self._backend_host = None
        self._backend_url = None
        self._http_proxy = None
        self._art_buffer_length = 5

    def backend_host(self, host: str = None):
        if host is not None:
            self._backend_host = host
        return self._backend_host

    def backend_url(self, url: str = None):
        if url is not None:
            self._backend_url = url
        return self._backend_url

    def art_buffer_length(self, length: int = None):
        if length is not None:
            if isinstance(length, int):
                self._art_buffer_length = length
        return self._art_buffer_length

    def http_proxy(self, proxy: str = None):
        if proxy is not None:
            self._http_proxy = proxy
        return self._http_proxy

    def init_from_environ(self):
        # dVars = {'APP_BACKEND_HOST': self.backend_host,  'APP_BACKEND_URL': self.backend_url,
        #          'APP_ART_BUFFER_LENGTH': self.art_buffer_length, 'http_proxy': self.http_proxy}

        dVars = {'APP_ART_BUFFER_LENGTH': self.art_buffer_length}

        for env_var in dVars.keys():
            env_var_value = get_env_variable(env_var)
            dVars[env_var](env_var_value)
