from .data import EXPECTED_CITATION_DATA, EXPECTED_PREDICTIONS, LISTWISE_AVG_RESULTS, PUB_MED_XML_INPUT_DATA
import gzip
import json
from mtix_descriptor_prediction_pipeline.pipeline import DescriptorPredictionPipeline, MedlineDateParser, MtiJsonResultsFormatter, PubMedXmlInputDataParser
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor
import os.path
from unittest import TestCase
from unittest.mock import MagicMock


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DATA_PATH = os.path.join(THIS_DIR, "data", "test_set_data.json.gz")
TEST_SET_PREDICTIONS_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Listwise22Avg_Results.json.gz")


class TestDescriptorPredictionPipeline(TestCase):
    
    def test_predict(self):
        medline_date_parser = MedlineDateParser()
        input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
        cnn_predictor = CnnModelTopNPredictor(100)
        pointwise_predictor = PointwiseModelTopNPredictor({}, 100)
        listwise_predictor = ListwiseModelTopNPredictor({}, 50)
        results_formatter = MtiJsonResultsFormatter({}, 0.475)
        pipeline = DescriptorPredictionPipeline(input_data_parser, cnn_predictor, pointwise_predictor, listwise_predictor, results_formatter)
        input_data = json.load(gzip.open(TEST_SET_DATA_PATH))
        expected_predictions = json.load(gzip.open(TEST_SET_PREDICTIONS_PATH))
        predictions = pipeline.predict(input_data)
        self.assertEqual(predictions, expected_predictions, "Predictions do not match expected result.")


class TestMedlineDateParser(TestCase):

    def assert_pub_year_extracted_correctly(self, text, expected_pub_year):
        parser = MedlineDateParser()
        pub_year = parser.extract_pub_year(text)
        self.assertEqual(pub_year, expected_pub_year, "Extracted wrong pub year")

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
        self.formatter = MtiJsonResultsFormatter({}, 0.475)

    def test_format(self):
        predictions = self.formatter.format(LISTWISE_AVG_RESULTS)
        self.assertEqual(predictions, EXPECTED_PREDICTIONS, "Predictions are not as expected.")


class TestPubMedXmlInputDataParser(TestCase):

    def setUp(self):
        self.medline_date_parser = MedlineDateParser()
        self.medline_date_parser.extract_pub_year = MagicMock(return_value=2021)
        self.parser = PubMedXmlInputDataParser(self.medline_date_parser)
    
    def test_parse_no_citations(self):
        input_data = []
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = []
        self.assertEqual(citaton_data, expected_citation_data, "Expected citation data to be an empty list.")

    def test_parse_one_citation(self):
        input_data = PUB_MED_XML_INPUT_DATA[:1]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = EXPECTED_CITATION_DATA[:1]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_two_citations(self):
        input_data = PUB_MED_XML_INPUT_DATA[:2]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = EXPECTED_CITATION_DATA[:2]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_citation_with_medline_date(self):
        input_data = [PUB_MED_XML_INPUT_DATA[2]]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = [EXPECTED_CITATION_DATA[2]]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")
        self.medline_date_parser.extract_pub_year.assert_called_once_with("2021 Mar-Apr 01")