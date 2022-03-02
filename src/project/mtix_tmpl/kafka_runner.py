import threading
import logging
import sys
import json

from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

from project.mtix_tmpl.app_config import cAppConfig
from project.mtix_tmpl.kafka_config import cKfkConfig
from project.mtix_tmpl.input_data import CInputData


logger = logging.getLogger(__name__)
logging.basicConfig(filename="stat.log", level="Info",
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class cKfkRunner(object):
    def __init__(self):
        self._cfg = cKfkConfig()
        self._consumer = None
        self._producer = None
        self._runner = None
        self._thread_id = None
        self._lock = threading.Lock()
        self._global_run = False
        self._stop_requested = False
        self._app_config = None
# -----------------------------------------------------------------------------------------------------------------------

    def __del__(self):
        # Close consumer
        if self._consumer is not None:
            try:
                self._consumer.close()
            except RuntimeError as re:
                msg = "Can't close consumer: \"{}\". Already closed?".format(str(re))
                print(msg, file=sys.stderr)

        # Close producer
        if self._producer is not None:
            try:
                self._producer.close()
            except RuntimeError as re:
                msg = "Can't close producer: \"{}\". Already closed?".format(str(re))
                print(msg, file=sys.stderr)
# -----------------------------------------------------------------------------------------------------------------------

    def runner(self, oRunner=None):
        if oRunner is not None:
            self._runner = oRunner
        return self._runner
# -----------------------------------------------------------------------------------------------------------------------

    def lock(self, oLock=None):
        if oLock is not None:
            self._lock = oLock
        return self._lock
# -----------------------------------------------------------------------------------------------------------------------

    def is_global_run(self):
        global_run = False
        with self.lock():
            global_run = self._global_run
        return global_run

    def set_global_run(self):
        with self.lock():
            self._global_run = True

    def clear_global_run(self):
        with self.lock():
            self._global_run = False
# -----------------------------------------------------------------------------------------------------------------------
    # Stop request related (a command from views)

    def is_stop_requested(self):
        stop_requested = False
        with self.lock():
            stop_requested = self._stop_requested
        return stop_requested

    def set_stop_requested(self):
        with self.lock():
            self._stop_requested = True

    def clear_stop_requested(self):
        with self.lock():
            self._stop_requested = False
# ----------------------------------------------------------------------------------------------------------------------

    # Saved thread ID
    def thread_id(self, thr_id: int = None):
        if thr_id is not None:
            try:
                self._thread_id = int(thr_id)
            except ValueError:
                msg = "Thread Id value must be integer. Obtained: \"{}\".".format(thr_id)
                print(msg, file=sys.stderr)
                raise Exception(msg)
        return self._thread_id
# -----------------------------------------------------------------------------------------------------------------------

    # Kafka config object
    def cfg(self, oCfg=None):
        if oCfg is not None:
            self._cfg = oCfg
        return self._cfg
# -----------------------------------------------------------------------------------------------------------------------

    # Kafka application config object
    def app_config(self, oCfg=None):
        if oCfg is not None:
            self._app_config = oCfg
        return self._app_config
# -----------------------------------------------------------------------------------------------------------------------

    # Kafka consumer object
    def consumer(self, oCons=None):
        if oCons is not None:
            self._consumer = oCons
        return self._consumer
# -----------------------------------------------------------------------------------------------------------------------

    # Kafka producer object
    def producer(self, oProd=None):
        if oProd is not None:
            self._producer = oProd
        return self._producer
    # -----------------------------------------------------------------------------------------------------------------------

    # Since thread can't be reused, create new Thread object
    def init_runner_thread(self):
        self.runner(threading.Thread(target=self.run, name='mtix_tmpl_runner', daemon=True))
# -----------------------------------------------------------------------------------------------------------------------

    # Commit changes (to move to the next article in kafka queue
    def consumer_commit(self, message):
        try:
            self.consumer().commit(message)
        except KafkaException as ke:
            print("Can't commit message: \"{}\".".format(str(ke)), file=sys.stderr)
            return False

        return True
# -----------------------------------------------------------------------------------------------------------------------

    # Restart consumer to avoid commit messages (in case of prediction errors or if it can't write results)
    def consumer_restart(self):
        # First stop consumer
        self.consumer().close()

        # Reconnect
        status, msg = self.initialize_kafka(is_producer=False)
        if not status:
            msg = "Consumer restart error: can't reconnect to Kafka: \"{}\"".format(msg)
            raise Exception(msg)
        else:
            print("Consumer restarted OK.")
    # -----------------------------------------------------------------------------------------------------------------------

    # Restart producer
    def producer_restart(self):
        # First stop producer
        self.producer().close()

        # Reconnect
        status, msg = self.initialize_kafka(is_consumer=False)
        if not status:
            msg = "Producer restart error: can't reconnect to Kafka: \"{}\"".format(msg)
            raise Exception(msg)
        else:
            print("Producer restarted OK.")
    # -----------------------------------------------------------------------------------------------------------------------

    def calculate_mesh_terms(self, lo_data_objects):
        # :param lo_articles: list of OBJECTS represented input data
        # Run models and make calculations here
        pass

        # Results are in this list
        l_results = list()

        # Now write result into Kafka producer topic
        try:
            self.notify(lo_data_objects)
        except KafkaException as ke:
            # Kafka related, try to restart producer. This could help.
            msg = "Kafka Producer Error: \"{}\". Restarting Kafka producer...".format(str(ke))
            print(msg, file=sys.stderr)

            # Try to restart Kafka producer
            try:
                self.producer_restart()
            except Exception as e:
                msg = "calculate_mesh_terms(): can't restart Kafka Producer: \"{}\".".format(str(e))
                print(msg, file=sys.stderr)
                return False

            # If we are here, producer was successfully restarted. Try one more time
            try:
                self.notify(lo_data_objects)
            except KafkaException as ke:
                # Kafka related, even producer restart did not help.
                msg = "Kafka Producer Error: \"{}\". Kafka producer restart did not help.".format(str(ke))
                print(msg, file=sys.stderr)
                return False

            # Everything is fine, restart helped
            return True
        except Exception as e:
            # Something else, probably severe
            msg = "Unknown error: can't put objects to \"{}\" topic. Kafka producer restart did not help." \
                  "".format(self.cfg().producer_topic(), str(e))
            print(msg, file=sys.stderr)
            return False

        return True
    # -----------------------------------------------------------------------------------------------------------------------

    def notify(self, l_results: list):
        if not isinstance(l_results, list):
            msg = "cKfkRunner::notify(): \"l_results\" parameter must be a list."
            raise Exception(msg)

        # Message headers, if you need them
        message_headers = {
            'desirable_header_1': "Desirable Header 1".encode("utf-8"),
            'desirable_header_N': "Desirable Header N".encode("utf-8"),
        }

        # Start write objects to results topic
        for o_result in l_results:
            self.producer().poll(0)
            try:
                # Convert result to bytes
                self.producer().produce(self.cfg().producer_topic(), str(o_result).encode('utf-8'),
                                        on_delivery=cKfkRunner.is_delivered, headers=message_headers)
            except BufferError as be:
                msg = "cKfkRunner::notify(): Buffer Error: \"{}\". Internal message queue is full?".format(str(be))
                raise KafkaException(msg)
            except KafkaException as ke:
                msg = "cKfkRunner::notify(): Kafka Exception: \"{}\".".format(str(ke))
                raise KafkaException(msg)
            except NotImplementedError as nie:
                msg = "cKfkRunner::notify(): NotImplemented Error: \"{}\".".format(str(nie))
                raise KafkaException(msg)
            except Exception as e:
                msg = "cKfkRunner::notify(): General Exception: \"{}\".".format(str(e))
                raise KafkaException(msg)

        # Check if something in still in the queue
        still_in_queue = self.producer().flush(self.cfg().flush_timeout())
        if still_in_queue > 0:
            msg = "Can't deliver message(s) to Kafka during timeout. Messages left in queue: {}".format(still_in_queue)
            raise KafkaException(msg)

        # Everything is fine
        msg = "cKfkRunner::notify(): all {} messages were delivered to \"{}\" topic." \
              "".format(len(l_results), self.cfg().producer_topic())
        print(msg)
    # -----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def is_delivered(err, msg):
        if err is not None:
            msg = "Failed to deliver message \"{}\" to topic \"{}\": \"{}\"." \
                  "".format(msg.value().decode('utf-8'), msg.topic(), str(err))
            print(msg, file=sys.stderr)
        else:
            print("Delivered message to topic {} partition [{}] @ offset {}"
                  "".format(msg.topic(), msg.partition(), msg.offset()))
# -----------------------------------------------------------------------------------------------------------------------

    # Thread content
    def run(self):

        print("Procedure RUN started.")
        # Clear stop request, if any
        self.clear_stop_requested()

        # Write thread id into object
        with self.lock():
            self.thread_id(threading.get_ident())

        # Articles are ready to process
        lObjects = list()
        MAX_LENGTH = self.app_config().art_buffer_length()

        # Main loop
        prev_message = None
        while True:
            try:
                if self.is_stop_requested():
                    print("Stop requested")
                    self.clear_stop_requested()
                    break

                message = None

                # Read message from the queue
                try:
                    message = self.consumer().poll(self.cfg().poll_timeout())
                except RuntimeError as re:
                    # Critical error here, may be kafka servers changed. Stop thread.
                    msg = "Runtime Error Exception: \"{}\". Consumer closed?.".format(str(re))
                    print(msg, file=sys.stderr)

                    # Try to reconnect (may be kafka brokers down?)
                    status, msg = self.initialize_kafka(is_producer=False)
                    if not status:
                        print("Can't reconnect to Kafka: \"{}\"".format(msg))
                        break
                    else:
                        print("Reconnected to Kafka OK.")
                except KafkaError as ke:
                    msg = "Kafka Error Exception: \"{}\". Internal error?".format(str(ke))
                    raise Exception(msg)
                except ValueError as ve:
                    msg = "Value Error Exception: \"{}\". May be num_messages > 1M?".format(str(ve))
                    raise Exception(msg)
                except Exception as ge:
                    msg = "General Exception: \"{}\"".format(str(ge))
                    raise Exception(msg)

                if message is None:
                    # No new articles in the queue. Process that were already caught
                    if len(lObjects) == 0:
                        continue

                    # Do with obtained data classes what need to be done
                    b_status = self.calculate_mesh_terms(lObjects)
                    lObjects.clear()

                    if not b_status:
                        # Avoid committing to Kafka if b_status is False
                        prev_message = None
                        continue

                    # Everything is fine, commit changes, move to the next object
                    if not self.consumer_commit(prev_message):
                        print("Restarting Kafka consumer...")
                        self.consumer_restart()
                    continue
                else:
                    prev_message = message

                err = message.error()
                if err:
                    msg = "An error occurred trying to read from broker: \"{}\". Ignored.".format(str(err))
                    print(msg, file=sys.stderr)
                    raise Exception(msg)

                print("Object was obtained from Kafka.")

                # "message.value()" contains input data as they were put into Kafka by data supplier
                try:
                    # Create an object from source data here
                    lo_input_records = cKfkRunner.from_kafka_message(message.value())
                    if len(lo_input_records) > 0:
                        lObjects.extend(lo_input_records)
                except Exception as e:
                    msg = "Can't obtain article XML: \"{}\". Omit it.".format(str(e))

                    # Commit changes: no necessity to pick up erroneous object
                    if not self.consumer_commit(prev_message):
                        msg += "Can't move from erroneous object to the next one."
                    print(msg, file=sys.stderr)
                    raise Exception(msg)

                # Process gathered articles if number of them exceeded maximum number
                if len(lObjects) < MAX_LENGTH:
                    continue

                # Maximum articles limit reached. Process buffer content
                b_status = self.calculate_mesh_terms(lObjects)
                lObjects.clear()

                if not b_status:
                    # Avoid committing to Kafka if b_status is False
                    prev_message = None
                    continue

                # Everything is fine, commit changes, move to the next object
                if not self.consumer_commit(prev_message):
                    print("Restarting Kafka consumer...")
                    self.consumer_restart()
                continue

            except Exception as ge:
                print(str(ge), file=sys.stderr)

        print("Thread is completed")
# -----------------------------------------------------------------------------------------------------------------------

    # Get Kafka message and prepare list of objects
    @staticmethod
    def from_kafka_message(b_message: bytes):
        l_batch_articles = list()

        d_articles = None
        try:
            d_articles = json.loads(b_message)
        except ValueError as ve:
            errmsg = "cKfkRunner::from_kafka_json(): Invalid JSON detected: \"{}\". What was obtained: \"{}\"." \
                     "".format(str(ve), b_message.decode('utf-8'))
            raise Exception(errmsg)

        # Valid JSON was obtained.from
        for d_art in d_articles:
            if not isinstance(d_art, dict):
                msg = "cKfkRunner::from_kafka_json(): entry is not a valid dictionary: \"{}\". Ignored.".format(str(d_art))
                print(msg, file=sys.stderr)
                continue

            # Make a class entry for each article
            try:
                oArt = CInputData.from_json(d_art)
                l_batch_articles.append(oArt)
            except Exception as e:
                msg = "cKfkRunner::from_kafka_json(): something wrong with article: \"{}\". Ignored.".format(str(e))
                print(msg, file=sys.stderr)
                continue

        return l_batch_articles
# -----------------------------------------------------------------------------------------------------------------------

    def initialize_kafka(self, is_producer: bool = True, is_consumer: bool = True):
        try:
            # Init object config from environment variables
            try:
                self.cfg().init_from_environ()
            except Exception as ge:
                msg = "Can't init config from environment: \"{}\". Exit.".format(str(ge))
                print(msg, file=sys.stderr)
                raise Exception(msg)

            # Generate a config for Kafka
            consumer_cfg, producer_cfg = None, None
            try:
                consumer_cfg = self.cfg().get_kafka_config()
                producer_cfg = self.cfg().get_kafka_config(is_producer=is_producer)
            except Exception as ge:
                msg = "Kafka config can't be initialized: \"{}\". Exit.".format(str(ge))
                print(msg, file=sys.stderr)
                raise Exception(msg)

            # Create Kafka Consumer and Producer
            if is_consumer:
                self.consumer(Consumer(consumer_cfg))
            if is_producer:
                self.producer(Producer(producer_cfg))

            # Subscribe on the topic
            if is_consumer:
                try:
                    self.consumer().subscribe([self.cfg().consumer_topic()])
                except KafkaException as ke:
                    msg = "Kafka Exception: \"{}\". Consumer closed?".format(str(ke))
                    print(msg, file=sys.stderr)
                    raise Exception(msg)

            # Create a runner thread
            if not self.check_thread():
                self.init_runner_thread()

            # Set flag that global runner is alive
            self.set_global_run()
        except Exception as ge:
            self.clear_global_run()
            print(str(ge), file=sys.stderr)
            return False, str(ge)

        return True, ""
# -----------------------------------------------------------------------------------------------------------------------

    # Thread starter
    def start(self):
        if self.runner() is not None:
            self.runner().start()
# -----------------------------------------------------------------------------------------------------------------------

    def check_thread(self):
        thread_id = None
        with self.lock():
            thread_id = self.thread_id()

        # print("Check Thread: thread_id is {}".format(thread_id))

        if thread_id is None:
            return False

        for thread in threading.enumerate():
            # print("Check Thread. Current thread: {}".format(thread.ident))
            if thread.ident == thread_id:
                return True
        return False
# -----------------------------------------------------------------------------------------------------------------------


# Read application environment
o_app_config = None
try:
    o_app_config = cAppConfig()
    o_app_config.init_from_environ()
except Exception as e:
    msg = "Application configuration error: \"{}\".".format(str(e))
    print(msg, file=sys.stderr)
    raise Exception(msg)


# Global variable of create Kafka wrapper
oGlobalKafkaRunner = cKfkRunner()

# Read and initiate from environment variables
oGlobalKafkaRunner.app_config(o_app_config)
