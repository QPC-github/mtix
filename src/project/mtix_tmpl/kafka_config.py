import re

import dns.resolver

from project.mtix_tmpl.kafka_utils import get_env_variable


class cKfkConfig():
    def __init__(self):
        self._mechanism = None
        self._protocol = None
        self._username = None
        self._password = None
        self._offset_reset = None
        self._auto_commit = None
        self._group_id = None
        self._client_id = None
        self._poll_timeout = 5
        self._flush_timeout = 2
        self._bootstrap_servers = None
        self._zone = None
        self._consumer_topic = None
        self._producer_topic = None

    def mechanism(self, mechanism: str = None):
        if mechanism is not None:
            self._mechanism = mechanism
        return self._mechanism

    def protocol(self, protocol: str = None):
        if protocol is not None:
            self._protocol = protocol
        return self._protocol

    def username(self, username: str = None):
        if username is not None:
            self._username = username
        return self._username

    def password(self, password: str = None):
        if password is not None:
            self._password = password
        return self._password

    def offset_reset(self, offset_reset: str = None):
        if offset_reset is not None:
            self._offset_reset = offset_reset
        return self._offset_reset

    def auto_commit(self, auto_commit: bool = None):
        if auto_commit is not None:
            self._auto_commit = auto_commit
        return self._auto_commit

    def group_id(self, group_id: str = None):
        if group_id is not None:
            self._group_id = group_id
        return self._group_id

    def client_id(self, client_id: str = None):
        if client_id is not None:
            self._client_id = client_id
        return self._client_id

    def bootstrap_servers(self, servers: str = None):
        if servers is not None:
            m = re.search(r'^(?:[^:]+:\d+,?)+', servers)
            if m is None:
                msg = "bootstrap_servers(): Bootstrap servers should be a comma-separated \"host:port\" values. " \
                      "Obtained: \"{}\". Stop.".format(servers)
                raise Exception(msg)
            else:
                self._bootstrap_servers = servers
        return self._bootstrap_servers

    def consumer_topic(self, topic: str = None):
        if topic is not None:
            self._consumer_topic = topic
        return self._consumer_topic

    def producer_topic(self, topic: str = None):
        if topic is not None:
            self._producer_topic = topic
        return self._producer_topic

    def poll_timeout(self, timeout: int = None):
        if timeout is not None:
            if isinstance(timeout, int):
                self._poll_timeout = timeout
        return self._poll_timeout

    def flush_timeout(self, timeout: int = None):
        if timeout is not None:
            if isinstance(timeout, int):
                self._flush_timeout = timeout
        return self._flush_timeout

    def zone(self, zone: str = None):
        if zone is not None:
            self._zone = zone
        return self._zone

    def init_from_environ(self):
        dVars = {'KFK_SASL_MECHANISM': self.mechanism,  'KFK_SECURITY_PROTOCOL': self.protocol,
                 'KFK_SASL_USERNAME': self.username,    'KFK_SASL_PASSWORD': self.password,
                 'KFK_OFFSET_RESET': self.offset_reset, 'KFK_AUTO_COMMIT': self.auto_commit,
                 'KFK_GROUP_ID': self.group_id,         'KFK_CLIENT_ID': self.client_id,
                 'KFK_CONSUMER_TOPIC': self.consumer_topic,
                 'KFK_PRODUCER_TOPIC': self.producer_topic,
                 'KFK_BOOTSTRAP_SERVERS': self.bootstrap_servers,
                 'KFK_ZONE': self.zone,
                 'KFK_POLL_TIMEOUT': self.poll_timeout, 'KFK_FLUSH_TIMEOUT':  self.flush_timeout
                 }

        for env_var in dVars.keys():
            env_var_value = get_env_variable(env_var)
            dVars[env_var](env_var_value)

        # Now cKfkConfig is initialized
# -----------------------------------------------------------------------------------------------------------------------
    def get_bootstrap(self):
        # Zone has higher priority
        if not self.zone():
            return self.bootstrap_servers()

        # Zone is available
        servers = []

        srv_records = dns.resolver.resolve(self.zone(), 'SRV')
        for srv in srv_records:
            record = "{}:{}".format(str(srv.target).rstrip('.'), srv.port)
            if record not in servers:
                servers.append(record)

        if len(servers) == 0:
            raise Exception("get_bootstrap(): no one host could be found for \"{}\".".format(self.zone()))

        hosts = ",".join(servers)
        print("get_bootstrap(): host list obtained from \"{}\" zone: \"{}\"".format(self.zone(), hosts))
        return hosts
# -----------------------------------------------------------------------------------------------------------------------

    def get_kafka_config(self, is_producer: bool = False):
        config = dict()

        mechanism = self.mechanism()
        if mechanism is not None and len(mechanism) > 0 and mechanism.lower() != 'none':
            config['sasl.mechanism'] = mechanism

        if self.protocol():
            config['security.protocol'] = self.protocol()

        username = self.username()
        if username is not None and len(username) > 0 and username.lower() != 'none':
            config['sasl.username'] = username

        password = self.password()
        if password is not None and len(password) > 0 and password.lower() != 'none':
            config['sasl.password'] = password

        if not is_producer:
            if self.offset_reset():
                config['auto.offset.reset'] = self.offset_reset()

        if not is_producer:
            if self.auto_commit():
                config['enable.auto.commit'] = self.auto_commit()

        if not is_producer:
            if self.group_id():
                config['group.id'] = self.group_id()

        if self.client_id():
            config['client.id'] = self.client_id()

        config['bootstrap.servers'] = self.get_bootstrap()

        return config
# -----------------------------------------------------------------------------------------------------------------------
