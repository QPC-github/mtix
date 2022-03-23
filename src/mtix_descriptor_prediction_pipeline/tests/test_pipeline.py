import json
from mtix_descriptor_prediction_pipeline.pipeline import DescriptorPredictionPipeline, InputDataParser, MtiJsonResultsFormatter
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor
from unittest import TestCase
#from unittest.mock import MagicMock

# TODO: Data should be included in package
TEST_SET_DATA_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_data/test_set_data.json"
TEST_SET_PREDICTIONS_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_predictions/test_set_2017-2022_Listwise22Avg_Results.json"


CITATION_DATA_1 = {"pmid": 32770536, 
                   "title": "Second Ventilatory Threshold Assessed by Heart Rate Variability in a Multiple Shuttle Run Test.",
                   "abstract": "Many studies have focused on heart rate variability in association with ventilatory thresholds. The purpose of the current study was to consider the ECG-derived respiration and the high frequency product of heart rate variability as applicable methods to assess the second ventilatory threshold (VT2). Fifteen healthy young soccer players participated in the study. Respiratory gases and ECGs were collected during an incremental laboratory test and in a multistage shuttle run test until exhaustion. VΤ2 was individually calculated using the deflection point of ventilatory equivalents. In addition, VT2 was assessed both by the deflection point of ECG-derived respiration and high frequency product. Results showed no statistically significant differences between VT2, and the threshold as determined with high frequency product and ECG-derived respiration (F(2,28)=0.83, p=0.45, η2=0.05). A significant intraclass correlation was observed for ECG-derived respiration (r=0.94) and high frequency product (r=0.95) with VT2. Similarly, Bland Altman analysis showed a considerable agreement between VT2 vs. ECG-derived respiration (mean difference of -0.06 km·h-1, 95% CL: ±0.40) and VT2 vs. high frequency product (mean difference of 0.02 km·h-1, 95% CL: ±0.38). This study suggests that, high frequency product and ECG-derived respiration are indeed reliable heart rate variability indices determining VT2 in a field shuttle run test.",
                   "journal_nlmid": "8008349",
                   "journal_name": "International journal of sports medicine",
                   "pub_year": 2021,
                   "year_completed": 2021}
CITATION_DATA_2 = {"pmid": 30455223,
                   "title": "Update on the biology and management of renal cell carcinoma.",
                   "abstract": "Renal cell cancer (RCC) (epithelial carcinoma of the kidney) represents 2%-4% of newly diagnosed adult tumors. Over the past 2 decades, RCC has been better characterized clinically and molecularly. It is a heterogeneous disease, with multiple subtypes, each with characteristic histology, genetics, molecular profiles, and biologic behavior. Tremendous heterogeneity has been identified with many distinct subtypes characterized. There are clinical questions to be addressed at every stage of this disease, and new targets being identified for therapeutic development. The unique characteristics of the clinical presentations of RCC have led to both questions and opportunities for improvement in management. Advances in targeted drug development and understanding of immunologic control of RCC are leading to a number of new clinical trials and regimens for advanced disease, with the goal of achieving long-term disease-free survival, as has been achieved in a proportion of such patients historically. RCC management is a promising area of ongoing clinical investigation.",
                   "journal_nlmid": "9501229", "jouranl_name": "Journal of investigative medicine : the official publication of the American Federation for Clinical Research",
                   "pub_year": 2019,
                   "year_completed": 2020,}


class TestDescriptorPredictionPipeline(TestCase):
    
    def test_predict(self):
        cnn_predictor = CnnModelTopNPredictor(100)
        pointwise_predictor = PointwiseModelTopNPredictor({}, 100)
        listwise_predictor = ListwiseModelTopNPredictor({}, 50)
        results_formatter = MtiJsonResultsFormatter({}, 0.475)
        pipeline = DescriptorPredictionPipeline(cnn_predictor, pointwise_predictor, listwise_predictor, results_formatter)
        input_data = json.load(open(TEST_SET_DATA_PATH))
        expected_predictions = json.load(open(TEST_SET_PREDICTIONS_PATH))
        predictions = pipeline.predict(input_data)
        self.assertEqual(predictions, expected_predictions, "Predictions do not match expected result.")


class TestInputDataParser(TestCase):

    def setUp(self):
        self.parser = InputDataParser()
        self.test_set_data = json.load(open(TEST_SET_DATA_PATH))
    
    def test_parse_no_citations(self):
        input_data = []
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = []
        self.assertEqual(citaton_data, expected_citation_data, "Expected citation data to be an empty list.")

    def test_parse_one_citation(self):
        input_data = self.test_set_data[:1]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = [CITATION_DATA_1,]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")

    def test_parse_two_citations(self):
        input_data = self.test_set_data[:2]
        citaton_data = self.parser.parse(input_data)
        expected_citation_data = [CITATION_DATA_1, CITATION_DATA_2]
        self.assertEqual(citaton_data, expected_citation_data, "Citation data different from expected citation data.")