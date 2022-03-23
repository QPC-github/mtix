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


def extract_citation_data(citation_xml):
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
        pub_year = extract_pub_year(medlinedate_text)
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


def extract_pub_year(medlinedate_text):
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