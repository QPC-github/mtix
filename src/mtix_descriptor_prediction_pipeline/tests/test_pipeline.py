import json
from mtix_descriptor_prediction_pipeline.pipeline import DescriptorPredictionPipeline, MedlineDateParser, PubMedXmlInputDataParser, MtiJsonResultsFormatter
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor
from unittest import TestCase
from unittest.mock import MagicMock

# TODO: Data should be included in package
TEST_SET_DATA_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_data/test_set_data.json"
TEST_SET_PREDICTIONS_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_predictions/test_set_2017-2022_Listwise22Avg_Results.json"


UNIT_TEST_DATA = [
    {
        "uid": 32770536,
        "data": "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"
    },
    {
        "uid": 30455223,
        "data": "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"
    },
    {
        "uid": 33449580,
        "data": "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"
    }
]

EXPECTED_CITATION_DATA=[
    {   
        "pmid": 32770536, 
        "title": "Second Ventilatory Threshold Assessed by Heart Rate Variability in a Multiple Shuttle Run Test.",
        "abstract": "Many studies have focused on heart rate variability in association with ventilatory thresholds. The purpose of the current study was to consider the ECG-derived respiration and the high frequency product of heart rate variability as applicable methods to assess the second ventilatory threshold (VT2). Fifteen healthy young soccer players participated in the study. Respiratory gases and ECGs were collected during an incremental laboratory test and in a multistage shuttle run test until exhaustion. VΤ2 was individually calculated using the deflection point of ventilatory equivalents. In addition, VT2 was assessed both by the deflection point of ECG-derived respiration and high frequency product. Results showed no statistically significant differences between VT2, and the threshold as determined with high frequency product and ECG-derived respiration (F(2,28)=0.83, p=0.45, η2=0.05). A significant intraclass correlation was observed for ECG-derived respiration (r=0.94) and high frequency product (r=0.95) with VT2. Similarly, Bland Altman analysis showed a considerable agreement between VT2 vs. ECG-derived respiration (mean difference of -0.06 km·h-1, 95% CL: ±0.40) and VT2 vs. high frequency product (mean difference of 0.02 km·h-1, 95% CL: ±0.38). This study suggests that, high frequency product and ECG-derived respiration are indeed reliable heart rate variability indices determining VT2 in a field shuttle run test.",
        "journal_nlmid": "8008349",
        "journal_title": "International journal of sports medicine",
        "pub_year": 2021,
        "year_completed": 2021,
    },
    {
        "pmid": 30455223,
        "title": "Update on the biology and management of renal cell carcinoma.",
        "abstract": "Renal cell cancer (RCC) (epithelial carcinoma of the kidney) represents 2%-4% of newly diagnosed adult tumors. Over the past 2 decades, RCC has been better characterized clinically and molecularly. It is a heterogeneous disease, with multiple subtypes, each with characteristic histology, genetics, molecular profiles, and biologic behavior. Tremendous heterogeneity has been identified with many distinct subtypes characterized. There are clinical questions to be addressed at every stage of this disease, and new targets being identified for therapeutic development. The unique characteristics of the clinical presentations of RCC have led to both questions and opportunities for improvement in management. Advances in targeted drug development and understanding of immunologic control of RCC are leading to a number of new clinical trials and regimens for advanced disease, with the goal of achieving long-term disease-free survival, as has been achieved in a proportion of such patients historically. RCC management is a promising area of ongoing clinical investigation.",
        "journal_nlmid": "9501229", 
        "journal_title": "Journal of investigative medicine : the official publication of the American Federation for Clinical Research",
        "pub_year": 2019,
        "year_completed": 2020,
    },
    {   
        "pmid": 33449580,
        "title": "\"HIV and Aging in Special Populations: From the Mitochondria to the Metropolis\"-Proceedings From the 2019 Conference.",
        "abstract": "",
        "journal_nlmid": "9111870",
        "journal_title": "The Journal of the Association of Nurses in AIDS Care : JANAC",
        "pub_year": 2021, 
        "year_completed": 2021, 
    },
]


class TestDescriptorPredictionPipeline(TestCase):
    
    def test_predict(self):
        medline_date_parser = MedlineDateParser()
        input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
        cnn_predictor = CnnModelTopNPredictor(100)
        pointwise_predictor = PointwiseModelTopNPredictor({}, 100)
        listwise_predictor = ListwiseModelTopNPredictor({}, 50)
        results_formatter = MtiJsonResultsFormatter({}, 0.475)
        pipeline = DescriptorPredictionPipeline(input_data_parser, cnn_predictor, pointwise_predictor, listwise_predictor, results_formatter)
        input_data = json.load(open(TEST_SET_DATA_PATH))
        expected_predictions = json.load(open(TEST_SET_PREDICTIONS_PATH))
        predictions = pipeline.predict(input_data)
        self.assertEqual(predictions, expected_predictions, "Predictions do not match expected result.")


class TestInputDataParser(TestCase):

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
        input_data = UNIT_TEST_DATA[:1]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = EXPECTED_CITATION_DATA[:1]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_two_citations(self):
        input_data = UNIT_TEST_DATA[:2]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = EXPECTED_CITATION_DATA[:2]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_citation_with_medline_date(self):
        input_data = [UNIT_TEST_DATA[2]]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = [EXPECTED_CITATION_DATA[2]]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")
        self.medline_date_parser.extract_pub_year.assert_called_once_with("2021 Mar-Apr 01")