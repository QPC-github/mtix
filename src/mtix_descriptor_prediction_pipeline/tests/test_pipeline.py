from mtix_descriptor_prediction_pipeline.pipeline import CitationDataSanitizer, DescriptorPredictionPipeline, MedlineDateParser, MtiJsonResultsFormatter, PubMedXmlInputDataParser
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTop100Predictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor
import os.path
from .test_data import *
from unittest import skip, TestCase
from unittest.mock import MagicMock, Mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DATA_PATH = os.path.join(THIS_DIR, "data", "test_set_data.json.gz")
TEST_SET_PREDICTIONS_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Listwise22Avg_Results.json.gz")

MAX_YEAR = 2021
THRESHOLD = 0.475

class TestCitationDataSanitizer(TestCase):

    def setUp(self):
        self.sanitizer = CitationDataSanitizer(MAX_YEAR)
        
    def test_sanitize_pass_through(self):
        citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": 2008,
                "year_completed": 2009,
            }
        }
        sanitized_citation_data = self.sanitizer.sanitize(citation_data)
        self.assertEqual(sanitized_citation_data, citation_data, "Expected no changes for complete citation data.")

    def test_sanitize_missing_journal_nlmid(self):
        citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": None,
                "journal_title": "The journal",
                "pub_year": 2008,
                "year_completed": 2009,
            }
        }
        expected_citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "<unknown>",
                "journal_title": "The journal",
                "pub_year": 2008,
                "year_completed": 2009,
            }
        }
        sanitized_citation_data = self.sanitizer.sanitize(citation_data)
        self.assertEqual(sanitized_citation_data, expected_citation_data, "Expected null journal nlmid to be replaced with <unknown>.")


    def test_sanitize_missing_pub_year(self):
        citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": None,
                "year_completed": 2009,
            }
        }
        expected_citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": 2009,
                "year_completed": 2009,
            }
        }
        sanitized_citation_data = self.sanitizer.sanitize(citation_data)
        self.assertEqual(sanitized_citation_data, expected_citation_data, "Expected null pub year to be replaced with year completed.")


    def test_sanitize_missing_year_completed(self):
        citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": 2008,
                "year_completed": None,
            }
        }
        expected_citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": 2008,
                "year_completed": 2021,
            }
        }
        sanitized_citation_data = self.sanitizer.sanitize(citation_data)
        self.assertEqual(sanitized_citation_data, expected_citation_data, "Expected null year completed to be replaced with 2021.")

    def test_sanitize_missing_pub_year_and_year_completed(self):
        citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": None,
                "year_completed": None,
            }
        }
        expected_citation_data = {
            123456789: {
                "pmid": 123456789,
                "title": "The title.",
                "abstract": "The abstract.",
                "journal_nlmid": "01234",
                "journal_title": "The journal",
                "pub_year": 2021,
                "year_completed": 2021,
            }
        }
        sanitized_citation_data = self.sanitizer.sanitize(citation_data)
        self.assertEqual(sanitized_citation_data, expected_citation_data, "Expected null pub year and year completed to be replaced by 2021.")


class TestDescriptorPredictionPipeline(TestCase):

    def setUp(self):
        medline_date_parser = MedlineDateParser()
        input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
        self.sanitizer = CitationDataSanitizer(MAX_YEAR)
        self.sanitizer.sanitize = Mock(wraps=self.sanitizer.sanitize)
        self.cnn_predictor = CnnModelTop100Predictor(None)
        self.cnn_predictor.predict = MagicMock(return_value=CNN_RESULTS)
        self.pointwise_predictor = PointwiseModelTopNPredictor(None, {}, 100)
        self.pointwise_predictor.predict = MagicMock(return_value=POINTWISE_RESULTS)
        self.listwise_predictor = ListwiseModelTopNPredictor(None, {}, 50)
        self.listwise_predictor.predict = MagicMock(return_value=LISTWISE_RESULTS)
        self.results_formatter = MtiJsonResultsFormatter(DESC_NAME_LOOKUP, DUI_LOOKUP, THRESHOLD)
        self.results_formatter.format = Mock(wraps=self.results_formatter.format)
        self.pipeline = DescriptorPredictionPipeline(input_data_parser, self.sanitizer, self.cnn_predictor, self.pointwise_predictor, self.listwise_predictor, self.results_formatter)

    def test_predict(self):
        input_data = PUBMED_XML_INPUT_DATA
        expected_predictions = EXPECTED_PREDICTIONS
        
        predictions = self.pipeline.predict(input_data)
        
        self.assertEqual(predictions, expected_predictions, "Predictions do not match expected result.")
        self.sanitizer.sanitize.assert_called_once_with(EXPECTED_CITATION_DATA)
        self.cnn_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA)
        self.pointwise_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA, CNN_RESULTS)
        self.listwise_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA, POINTWISE_AVG_RESULTS)
        self.results_formatter.format.assert_called_once_with(UNORDERED_LISTWISE_AVG_RESULTS)


class TestMedlineDateParser(TestCase):

    def assert_pub_year_extracted_correctly(self, text, expected_pub_year):
        parser = MedlineDateParser()
        pub_year = parser.extract_pub_year(text)
        self.assertEqual(pub_year, expected_pub_year, "Extracted wrong pub year.")

    def test_extract_pub_year(self):
        self.assert_pub_year_extracted_correctly("2021 Mar-Apr 01", 2021)
        self.assert_pub_year_extracted_correctly("1998 Dec-1999 Jan", 1998)
        self.assert_pub_year_extracted_correctly("2022 Spring", 2022)
        self.assert_pub_year_extracted_correctly("2016 Spring-Summer", 2016)
        self.assert_pub_year_extracted_correctly("1965 Nov-Dec", 1965)
        self.assert_pub_year_extracted_correctly("2000 Dec 23-30", 2000)
        self.assert_pub_year_extracted_correctly("", None)
        self.assert_pub_year_extracted_correctly("invalid", None)
        self.assert_pub_year_extracted_correctly("Summer 2009", 2009)
        self.assert_pub_year_extracted_correctly("24th March 2018", 2018)
        self.assert_pub_year_extracted_correctly("24th Mar '01", 2001)


class TestMtiJsonResultsFormatter(TestCase):

    def setUp(self):
        self.formatter = MtiJsonResultsFormatter(DESC_NAME_LOOKUP, DUI_LOOKUP, THRESHOLD)

    def test_format(self):
        predictions = self.formatter.format(UNORDERED_LISTWISE_AVG_RESULTS)
        self.assertEqual(predictions, EXPECTED_PREDICTIONS, "Predictions are not as expected.")


class TestPubMedXmlInputDataParser(TestCase):

    def setUp(self):
        self.medline_date_parser = MedlineDateParser()
        self.medline_date_parser.extract_pub_year = MagicMock(return_value=2021)
        self.parser = PubMedXmlInputDataParser(self.medline_date_parser)
    
    def test_parse_no_citations(self):
        input_data = []
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = {}
        self.assertEqual(citaton_data, expected_citation_data, "Expected citation data to be an empty dictionary.")

    def test_parse_one_citation(self):
        input_data = [PUBMED_XML_INPUT_DATA[0]]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = { 32770536: EXPECTED_CITATION_DATA[32770536]}
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_two_citations(self):
        input_data = PUBMED_XML_INPUT_DATA
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = EXPECTED_CITATION_DATA
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_citation_with_medline_date(self):
        input_data = MEDLINEDATE_PUBMED_XML_INPUT_DATA
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = MEDLINEDATE_EXPECTED_CITATION_DATA
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")
        self.medline_date_parser.extract_pub_year.assert_called_once_with("2021 Mar-Apr 01")