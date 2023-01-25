import base64
import dateutil.parser
import pandas as pd
import re
import xml.etree.ElementTree as ET
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


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


class Base64Helper:

    def encode(self, text, level=-1):
        text = text.encode(ENCODING)
        text = zlib.compress(text, level=level)
        text = base64.b64encode(text) 
        text = text.decode(ENCODING) 
        return text

    def decode(self, text):
        text = base64.b64decode(text) 
        text = zlib.decompress(text)
        text = text.decode(ENCODING)
        return text


class CitationDataSanitizer:

    def __init__(self, max_year):
        self.max_year = max_year
        self.min_year_completed = 1965
        self.min_pub_year = 1902
        
    def sanitize(self, citation_data):
        if citation_data["journal_nlmid"] is None:
            citation_data["journal_nlmid"] = "<unknown>"
            
        if citation_data["pub_year"] is None:
            if citation_data["year_completed"] is not None:
                citation_data["pub_year"] = citation_data["year_completed"]
            else:
                citation_data["pub_year"] = self.max_year
        if citation_data["year_completed"] is None:
            citation_data["year_completed"] = self.max_year

        citation_data["pub_year"] = min(self.max_year, citation_data["pub_year"])
        citation_data["pub_year"] = max(self.min_pub_year, citation_data["pub_year"])

        citation_data["year_completed"] = min(self.max_year, citation_data["year_completed"])
        citation_data["year_completed"] = max(self.min_year_completed, citation_data["year_completed"])

    def sanitize_list(self, citation_data_list):
        for citation_data in citation_data_list:
            self.sanitize(citation_data)


class PubMedXmlInputDataParser:
    def __init__(self):
        self.base64_helper = Base64Helper()
        medline_date_parser = MedlineDateParser()
        self.xml_parser = PubMedXmlParser(medline_date_parser)
       
    def parse(self, input_data):
        if type(input_data) is not list:
            raise ValueError("Input data must be a list.")

        citation_data_list = []
        for item in input_data:
            data = item["data"]
            citation_data = self.parse_data(data)
            citation_data_list.append(citation_data)
        return citation_data_list

    def parse_data(self, data):
        citation_xml = self.base64_helper.decode(data)
        citation_data = self.xml_parser.parse(citation_xml)
        return citation_data


class PubMedXmlParser:
    def __init__(self, medline_date_parser):
        self.medline_date_parser = medline_date_parser

    def parse(self, citation_xml):
        medline_citation_node = ET.fromstring(citation_xml)

        pmid_node = medline_citation_node.find("PMID")
        pmid = pmid_node.text.strip()
        pmid = int(pmid)

        title = ""
        title_node = medline_citation_node.find("Article/ArticleTitle") 
        title_text = ET.tostring(title_node, encoding="unicode", method="text")
        if title_text is not None:
            title = title_text.strip()
                
        abstract = ""
        abstract_node = medline_citation_node.find("Article/Abstract")
        if abstract_node is not None:
            abstract_text_node_list = abstract_node.findall("AbstractText")
            for abstract_text_node in abstract_text_node_list:
                if "Label" in abstract_text_node.attrib:
                    if len(abstract) > 0:
                        abstract += " "
                    abstract += abstract_text_node.attrib["Label"].strip() + ": "
                abstract_text = ET.tostring(abstract_text_node, encoding="unicode", method="text")
                if abstract_text is not None:
                    abstract += abstract_text.strip()

        journal_nlmid = None
        journal_nlmid_node = medline_citation_node.find("MedlineJournalInfo/NlmUniqueID")
        if journal_nlmid_node is not None:
            journal_nlmid = journal_nlmid_node.text.strip()

        journal_title = ""
        journal_title_node = medline_citation_node.find("Article/Journal/Title")
        if journal_title_node is not None:
            journal_title_text = ET.tostring(journal_title_node, encoding="unicode", method="text")
            if journal_title_text is not None:
                journal_title = journal_title_text.strip() 

        pub_year = None
        medline_date_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date_node is not None:
            medline_date_text = ET.tostring(medline_date_node, encoding="unicode", method="text")
            if medline_date_text is not None:
                medline_date_text = medline_date_text.strip()
                pub_year = self.medline_date_parser.extract_pub_year(medline_date_text)
        else:
            pub_year_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/Year")
            pub_year = pub_year_node.text.strip()
            pub_year = int(pub_year)

        year_completed = None
        date_completed_node = medline_citation_node.find("DateCompleted")
        if date_completed_node is not None:
            year_completed_node = date_completed_node.find("Year")
            year_completed = year_completed_node.text.strip()
            year_completed = int(year_completed)
        
        citation_data = {
                    "pmid": pmid, # int, Not None
                    "title": title, # str Not None, possibly ""
                    "abstract": abstract, # str Not None, possibly ""
                    "journal_nlmid": journal_nlmid, # str, Possibly None
                    "journal_title": journal_title, # str, Not None, possibly ""
                    "pub_year": pub_year, # int, Possibly None
                    "year_completed": year_completed, # int, Possibly None
                    }

        return citation_data


class MedlineDateParser:
    def extract_pub_year(self, text):
        pub_year = text[:4]
        try:
            pub_year = int(pub_year)
        except ValueError:
            match = re.search(r"\d{4}", text)
            if match:
                pub_year = match.group(0)
                pub_year = int(pub_year)
            else:
                try:
                    pub_year = dateutil.parser.parse(text, fuzzy=True).date().year
                except:
                    pub_year = None
        return pub_year