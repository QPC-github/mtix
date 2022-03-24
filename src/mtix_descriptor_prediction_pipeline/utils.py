import base64
import pandas as pd
import zlib


ENCODING = "utf-8"


def avg_top_results(input_top_results, output_top_results):
    average_top_results = []
    return average_top_results


def base64_decode(text):
    text = base64.b64decode(text) 
    text = zlib.decompress(text)
    text = text.decode("utf-8")
    return text


def base64_encode(text, level=-1):
    text = text.encode(ENCODING)
    text = zlib.compress(text, level=level)
    text = base64.b64encode(text) 
    text = text.decode(ENCODING) 
    return text


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def create_query_lookup(citation_data): # Is this the right place for this function
    query_lookup = None
    return query_lookup