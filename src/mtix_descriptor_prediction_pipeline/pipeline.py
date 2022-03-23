import dateutil.parser
import re
from .utils import avg_top_results, base64_decode, create_query_lookup
import xml.etree.ElementTree as ET


class DescriptorPredictionPipeline:
    def __init__(self, input_data_parser, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter):
        self.input_data_parser = input_data_parser
        self.cnn_model_top_n_predictor = cnn_model_top_n_predictor
        self.pointwise_model_top_n_predictor = pointwise_model_top_n_predictor
        self.listwise_model_top_n_predictor = listwise_model_top_n_predictor
        self.results_formatter = results_formatter

    def predict(self, input_data):
        citation_data = self.input_data_parser.parse(input_data)
        query_lookup = create_query_lookup(citation_data)
        cnn_results = self.cnn_model_top_n_predictor.predict(citation_data)
        pointwise_results = self.pointwise_model_top_n_predictor.predict(query_lookup, cnn_results)
        pointwsie_avg_results = avg_top_results(cnn_results, pointwise_results)
        listwise_results = self.listwise_model_top_n_predictor.predict(query_lookup, pointwsie_avg_results)
        listwise_avg_results = avg_top_results(pointwsie_avg_results, listwise_results)
        predictions = self.results_formatter.format(listwise_avg_results)
        return predictions


class MedlineDateParser:
    def extract_pub_year(self, medlinedate_text):
        pub_year = medlinedate_text[:4]
        try:
            pub_year = int(pub_year)
        except ValueError:
            match = re.search(r"\d{4}", medlinedate_text)
            if match:
                pub_year = match.group(0)
                pub_year = int(pub_year)
            else:
                try:
                    pub_year = dateutil.parser.parse(medlinedate_text, fuzzy=True).date().year
                except:
                    pub_year = None
        return pub_year


class PubMedXmlInputDataParser:
    def __init__(self, medline_date_parser):
        self.medline_date_parser = medline_date_parser
    
    def parse(self, input_data):
        citation_data_list = []
        for item in input_data:
            citation_xml = item["data"]
            citation_xml = base64_decode(citation_xml)
            citation_data = self._parse_xml(citation_xml)
            citation_data_list.append(citation_data)
        return citation_data_list

    def _parse_xml(self, citation_xml):
        medline_citation_node = ET.fromstring(citation_xml)

        pmid_node = medline_citation_node.find("PMID")
        pmid = pmid_node.text.strip()
        pmid = int(pmid)

        title = ""
        title_node = medline_citation_node.find("Article/ArticleTitle") 
        title = ET.tostring(title_node, encoding="unicode", method="text")
        title = title.strip() if title is not None else ""
        
        abstract = ""
        abstract_node = medline_citation_node.find("Article/Abstract")
        if abstract_node is not None:
            abstract_text_nodes = abstract_node.findall("AbstractText")
            for abstract_text_node in abstract_text_nodes:
                if "Label" in abstract_text_node.attrib:
                    if len(abstract) > 0:
                        abstract += " "
                    abstract += abstract_text_node.attrib["Label"].strip() + ": "
                abstract_text = ET.tostring(abstract_text_node, encoding="unicode", method="text")
                if abstract_text is not None:
                    abstract += abstract_text.strip()

        journal_nlmid_node = medline_citation_node.find("MedlineJournalInfo/NlmUniqueID")
        journal_nlmid = journal_nlmid_node.text.strip() if journal_nlmid_node is not None else None

        journal_title_node = medline_citation_node.find("Article/Journal/Title")
        journal_title = ""
        if journal_title_node is not None:
            journal_title = ET.tostring(journal_title_node, encoding="unicode", method="text")
            journal_title = journal_title.strip() if journal_title is not None else ""

        medlinedate_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medlinedate_node is not None:
            medlinedate_text = medlinedate_node.text.strip()
            pub_year = self.medline_date_parser.extract_pub_year(medlinedate_text)
        else:
            pub_year_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/Year")
            pub_year = pub_year_node.text.strip()
            pub_year = int(pub_year)

        year_completed = None
        date_completed_node = medline_citation_node.find("DateCompleted")
        if date_completed_node is not None:
            year_completed = int(date_completed_node.find("Year").text.strip())
        
        citation_data = {
                    "pmid": pmid, 
                    "title": title, 
                    "abstract": abstract,
                    "journal_nlmid": journal_nlmid,
                    "journal_title": journal_title,
                    "pub_year": pub_year,
                    "year_completed": year_completed,
                    }

        return citation_data


class MtiJsonResultsFormatter:
    def __init__(self, dui_lookup, threshold):
        self.dui_lookup = dui_lookup
        self.threshold = threshold

    def format(self, results):
        mti_json_object = None
        return mti_json_object