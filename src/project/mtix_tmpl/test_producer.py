#!/usr/bin/env python

from confluent_kafka import Producer, KafkaException
import json
import dns.resolver


def delivery_check(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to \"{}\" [{}]'.format(msg.topic(), msg.partition()))
# -----------------------------------------------------------------------------------------------------------------------


def ReadJson(file_name: str):
    fh = None
    try:
        fh = open(file_name, mode='rt', encoding='utf-8')
    except IOError as ioe:
        raise Exception("Can't read source JSON: \"{}\". File: \"{}\".".format(str(ioe), file_name))

    content = fh.read()
    fh.close()

    return content
# -----------------------------------------------------------------------------------------------------------------------


def PrepareJson():
    dEvent = {
       "eventVersion": "2.1",
       "eventSource": "aws:s3",
       "awsRegion": "us-east-1",
       "eventTime": "2020-09-02T19:17:25.713Z",
       "eventName": "ObjectCreated:CompleteMultipartUpload",
       "userIdentity": {
          "principalId": "AWS:AROATUZNIXZYGRNLCTIIW:timonin@ncbi.nlm.nih.gov"
       },
       "requestParameters": {
          "sourceIPAddress": "130.14.25.182"
       },
       "responseElements": {
          "x-amz-request-id": "CEF550A6C61B5219",
          "x-amz-id-2": "5GR7he7jkbzr75ULoMMoTL3DIQoG8LsutjaOLga6gi3DaZAMHhdprZDCp9gQ9zZ+GvU6YpJNqt7arGDaguo1alT9RQVD1csv"
       },
       "s3": {
          "s3SchemaVersion": "1.0",
          "configurationId": "4d2a19b2-5926-4c89-a2b9-4b2a78d1f3d4",
          "bucket": {
             "name": "timonin-nidx-src",
             "ownerIdentity": {
                "principalId": "AD1Z42EX7ZW85"
             },
             "arn": "arn:aws:s3:::timonin-nidx-src"
          },
          "object": {
             "key": "nuccore/xml-8-5-2020/uidterms.xml.10",
             "size": 166468107,
             "eTag": "108b80b223de3eaf8e95f8e7f080652a-10",
             "sequencer": "005F4FEFBF18175B69"
          }
       }
    }

    event_json = None
    try:
        event_json = json.JSONEncoder().encode(dEvent)
    except Exception:
        return None

    return event_json
# -----------------------------------------------------------------------------------------------------------------------


servers = []
srv_records = dns.resolver.resolve('_kafka_ssl._tcp.dev.ac-va.ncbi.nlm.nih.gov', 'SRV')
for srv in srv_records:
    record = "{}:{}".format(str(srv.target).rstrip('.'), srv.port)
    servers.append(record)

topic_name = 'timdev'

config = {'bootstrap.servers': ",".join(servers),
          'security.protocol': 'SSL',
          'client.id': 'mtix-tmpl-client',
          }

producer = Producer(config)

# jevent = PrepareJson()
jevent = ReadJson("/home/timonin/work/django/mtix/pubmed22n1212.json")

dHeaders = {"bucket": "timonin-nidx-src", "file": "nuccore/xml-8-5-2020/uidterms.xml.20",
            "ingest": "s3://timonin-nidx-dst/ingest",   "redshift": "/nuccore/redshift"}

for event in [jevent]:
    producer.poll(0)

    try:
        producer.produce(topic_name, event.encode("utf-8"), on_delivery=delivery_check, headers=dHeaders)
    except BufferError as be:
        msg = "Buffer Error: \"{}\". Internal message queue is probably full.".format(str(be))
        print(msg)
        raise Exception(msg)
    except KafkaException as ke:
        msg = "Kafka Exception: \"{}\".".format(str(ke))
        print(msg)
        raise Exception(msg)
    except NotImplementedError as nie:
        msg = "NotImplemented Error: \"{}\".".format(str(nie))
        print(msg)
        raise Exception(msg)
    except Exception as e:
        msg = "General Exception: \"{}\".".format(str(e))
        print(msg)
        raise Exception(msg)

q = producer.flush()
# -----------------------------------------------------------------------------------------------------------------------
