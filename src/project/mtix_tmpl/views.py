from django.views.generic import View
from django.http import HttpResponse

import logging

import os
import sys
import subprocess

from project.mtix_tmpl.kafka_runner import oGlobalKafkaRunner

logger = logging.getLogger(__name__)


class cvHealthcheck(View):

    def __init__(self):
        super(cvHealthcheck, self).__init__()

    def get(self, request):

        if not oGlobalKafkaRunner.is_global_run():
            status, msg = oGlobalKafkaRunner.initialize_kafka()
            if not status:
                return HttpResponse("Can't connect to kafka: \"{}\"".format(msg), status=500)

        # Kafka Runner is already prepared. Check thread
        if oGlobalKafkaRunner.check_thread():
            return HttpResponse('200 OK', status=200)

        # No thread exists. Try to start
        try:
            oGlobalKafkaRunner.start()
        except Exception:

            # Thread start failed. One more attempt with new global runner
            status, msg = oGlobalKafkaRunner.initialize_kafka()
            if not status:
                return HttpResponse("Can't start thread, reconnect to kafka failed", status=500)
            else:
                try:
                    oGlobalKafkaRunner.start()
                except Exception:
                    return HttpResponse("Can't start thread even after successful kafka restart", status=500)

        return HttpResponse('200 OK, thread (re)started', status=200)
# -----------------------------------------------------------------------------------------------------------------------


class cStopKafka(View):
    def __init__(self):
        super(cStopKafka, self).__init__()

    def get(self, request):
        if oGlobalKafkaRunner.check_thread():
            oGlobalKafkaRunner.set_stop_requested()

            # Finish this thread
            oGlobalKafkaRunner.runner().join()

        return HttpResponse('Kafka Stop has successfully been requested')
# -----------------------------------------------------------------------------------------------------------------------


class cKafkaEnv(View):
    def __init__(self):
        super(cKafkaEnv, self).__init__()

    def get(self, request):

        # Report
        report = "Environment variables:\n\n"

        # Environment variables
        for env, value in dict(os.environ).items():
            report += "{}={}\n".format(env, value)

        # Get python and pip
        python = sys.executable
        pip = os.path.join(os.path.split(python)[0], "pip")

        report += "\nPython binary via \"sys.executable\": \"{}\"\n".format(python)

        # Python version via "--version"
        cmd = "{} --version".format(python)
        reply = None
        try:
            reply = run_subprocess(cmd)
            report += "Python version via \"--version\": {}".format(reply)
        except Exception as ex:
            report += "Can't get Python version: \"{}\". Command: \"{}\".\n".format(str(ex), cmd)

        report += "Python version via \"sys.version\": {}\n".format(sys.version)

        # Python modules
        cmd = "{} freeze".format(pip)
        try:
            reply = run_subprocess(cmd)
            report += "\nModules installed: \n{}\n".format(reply)
        except Exception as ex:
            report += "Can't get Python version: \"{}\", command: \"{}\".\n".format(str(ex), cmd)

        return HttpResponse(report, content_type='text/plain')
# -----------------------------------------------------------------------------------------------------------------------


def run_subprocess(cmd):
    exhaust = None

    try:
        exhaust = subprocess.check_output(cmd, shell=True).decode("utf-8")
    except OSError as ose:
        errmsg = "OS Error Exception: \"{}\". Command: \"{}\".".format(str(ose), cmd)
        raise Exception(errmsg)
    except subprocess.CalledProcessError as cpe:
        errmsg = "CalledProcessError Exception: \"{}\". Code: {}. Output: \"{}\". " \
                 "Command: \"{}\".".format(str(cpe), cpe.returncode, cpe.output, cmd)
        raise Exception(errmsg)
    except Exception as e:
        errmsg = "General Exception: \"{}\". Command: \"{}\".".format(str(e), cmd)
        raise Exception(errmsg)

    return exhaust
# -----------------------------------------------------------------------------------------------------------------------
