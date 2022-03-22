import base64
import gzip
import json
import os.path
import requests
import xml.etree.ElementTree as ET
import zlib


ENCODING="utf-8"
LIMIT = 1000000000
MAX_RETRYS = 3
PUBONE_URL = "https://pubmed.ncbi.nlm.nih.gov/api/pubone/pubmed/pubmed_"
WORKING_DIR="/home/raear/working_dir/mtix/scripts"


def base64_encode(text):
    text = text.encode(ENCODING)
    text = zlib.compress(text)
    text = base64.b64encode(text) 
    text = text.decode(ENCODING) 
    return text

def get_pubone_xml(pmid):
    url = PUBONE_URL + str(pmid)
    retrys = 0
    while True:
        try:
            pubone_xml = requests.get(url).content
            root_node = ET.fromstring(pubone_xml)
            break
        except ET.ParseError as e:
            print(e)
            retrys +=1
            if retrys < MAX_RETRYS:
                continue
            else:
                raise Exception("Exceeded max retrys.")
    return root_node

def main():
    test_set_path = os.path.join(WORKING_DIR, "test_set.jsonl.gz")
    test_set_data_path = os.path.join(WORKING_DIR, "test_set_data.json")

    test_set_data = []
    with gzip.open(test_set_path, "rt", encoding=ENCODING) as file:
        for idx, line in enumerate(file.readlines()[:LIMIT]):
            print(idx, end="\r")
            citation_data = json.loads(line.strip())
            pmid = citation_data["pmid"]
            root_node = get_pubone_xml(pmid)
            medline_citation_node = root_node.find("PubmedArticle/MedlineCitation")
            medline_citation_node_xml = ET.tostring(medline_citation_node, encoding=ENCODING, method="xml").decode(ENCODING)
            medline_citation_node_xml = base64_encode(medline_citation_node_xml)
            test_set_data.append({ "uid": pmid, "data": medline_citation_node_xml })

    json.dump(test_set_data, open(test_set_data_path, "wt", encoding=ENCODING), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()