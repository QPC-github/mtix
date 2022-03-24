import base64
import pandas as pd
import zlib


ENCODING = "utf-8"


def average_top_results(input_top_results, output_top_results):
    result_list = [output_top_results, input_top_results]
    results_count = len(result_list)
    
    average_results = {}
    for q_id in result_list[0]:
        average_results[q_id] = {}
        for p_id in result_list[0][q_id]:
            scores = [results[q_id][p_id] for results in result_list]
            avg = sum(scores) / results_count
            average_results[q_id][p_id] = avg
    return average_results


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