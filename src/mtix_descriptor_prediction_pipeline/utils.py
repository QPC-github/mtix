import base64
import dateutil.parser
import zlib
import re
import xml.etree.ElementTree as ET


ENCODING = "utf-8"


def base64_decode(text):
    text = base64.b64decode(text) 
    text = zlib.decompress(text)
    text = text.decode("utf-8")
    return text


def base64_encode(text):
    text = text.encode(ENCODING)
    text = zlib.compress(text)
    text = base64.b64encode(text) 
    text = text.decode(ENCODING) 
    return text


def avg_top_results(input_top_results, output_top_results):
    average_top_results = None
    return average_top_results


def create_query_lookup(citation_data): # Is this the right place for this function
    query_lookup = None
    return query_lookup