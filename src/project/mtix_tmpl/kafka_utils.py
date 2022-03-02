import os
import re
import logging

logger = logging.getLogger(__name__)


def get_env_variable(env_var):
    value = None

    try:
        value = os.environ[env_var]
    except KeyError:
        errmsg = "Environment variable \"{}\" does not exist. Stop.".format(env_var)
        raise Exception(errmsg)

    if value is None or value.lower() == "none":
        return None

    # Adjust values
    m = re.search(r'^\s*(\d+)\s*$', value)
    if m is not None:
        value = int(m.group(1))
    else:
        m = re.search(r'\s*(true|false)\s*$', value, re.I)
        if m is not None:
            value = True if m.group(1).lower() == 'true' else False

    return value
# -----------------------------------------------------------------------------------------------------------------------
