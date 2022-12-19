from .data import *
import gzip
import json
from mtix import create_async_pipeline
from mtix.sagemaker_factory import create_descriptor_prediction_pipeline
import os.path
import pytest
from .utils import compute_metrics, TestCaseBase


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
VPC_ENDPOINT = None
DESC_NAME_LOOKUP_PATH =                                 os.path.join(DATA_DIR, "main_heading_names.tsv")
DUI_LOOKUP_PATH =                                       os.path.join(DATA_DIR, "main_headings.tsv")
SUBHEADING_NAME_LOOKUP_PATH =                           os.path.join(DATA_DIR, "subheading_names.tsv")
TEST_SET_DATA_PATH =                                    os.path.join(DATA_DIR, "test_set_data.json.gz")
TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH =                 os.path.join(DATA_DIR, "test_set_2017-2022_Descriptor_Ground_Truth.json.gz")
TEST_SET_EXPECTED_CHAINED_SUBHEADING_PREDICTIONS_PATH = os.path.join(DATA_DIR, "test_set_2017-2022_Chained_Subheading_Predictions.json.gz")
TEST_SET_EXPECTED_DESCRIPTOR_PREDICTIONS_PATH =         os.path.join(DATA_DIR, "test_set_2017-2022_Listwise22Avg_Results.json.gz")
TEST_SET_SUBHEADING_GROUND_TRUTH_PATH =                 os.path.join(DATA_DIR, "test_set_2017-2022_Subheading_Ground_Truth.json.gz")

@pytest.mark.integration
class TestDescriptorPredictionPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_descriptor_prediction_pipeline(DESC_NAME_LOOKUP_PATH, 
                                            DUI_LOOKUP_PATH, 
                                            "raear-cnn-endpoint-2022-v1-async", 
                                            "raear-pointwise-endpoint-2022-v2-async", 
                                            "raear-listwise-endpoint-2022-v2-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128,
                                            vpc_endpoint=VPC_ENDPOINT)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_first_five_articles(self):
        limit = 5
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_DESCRIPTOR_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
        predictions = self._predict(limit)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(gzip.open(TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)
        precision, recall, f1score = compute_metrics(ground_truth, predictions)
        
        self.assertAlmostEqual(f1score,   0.6955, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected.")
        self.assertAlmostEqual(precision, 0.7238, delta=delta, msg=f"Precision of {precision:.4f} not as expected.")
        self.assertAlmostEqual(recall,    0.6694, delta=delta, msg=f"Recall of {recall:.4f} not as expected.")
        
    def test_replace_brackets(self):
        # PMID 33998125 contains an abstract with the pattern "] [".
        self.pipeline.predict(ARTICLE_33998125)


@pytest.mark.integration
class TestIndexingPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_async_pipeline(DESC_NAME_LOOKUP_PATH, 
                                            DUI_LOOKUP_PATH, 
                                            SUBHEADING_NAME_LOOKUP_PATH,
                                            "raear-cnn-endpoint-2022-v1-async", 
                                            "raear-pointwise-endpoint-2022-v2-async", 
                                            "raear-listwise-endpoint-2022-v2-async",
                                            "raear-all-subheading-cnn-endpoint-2022-v1-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128,
                                            subheading_batch_size=128,
                                            vpc_endpoint=VPC_ENDPOINT)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_first_five_articles(self):
        limit = 5
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_CHAINED_SUBHEADING_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
        predictions = self._predict(limit)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(gzip.open(TEST_SET_SUBHEADING_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)

        precision, recall, f1score = compute_metrics(ground_truth, predictions)
        self.assertAlmostEqual(f1score,   0.4731, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(precision, 0.4679, delta=delta, msg=f"Precision of {precision:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(recall,    0.4784, delta=delta, msg=f"Recall of {recall:.4f} not as expected for all subheadings.")

        precision, recall, f1score = compute_metrics(ground_truth, predictions, CRITICAL_SUBHEADINGS)
        self.assertAlmostEqual(f1score,   0.4905, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(precision, 0.4839, delta=delta, msg=f"Precision of {precision:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(recall,    0.4972, delta=delta, msg=f"Recall of {recall:.4f} not as expected for critical subheadings.")