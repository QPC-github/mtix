import gzip
import json
import math
from mtix import create_async_descriptor_prediction_pipeline
import os.path
import pytest
from unittest import skip, TestCase
from .data import *


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESC_NAME_LOOKUP_PATH = os.path.join(THIS_DIR, "data", "main_heading_names.tsv")
DUI_LOOKUP_PATH = os.path.join(THIS_DIR, "data", "main_headings.tsv")
SUBHEADING_NAME_LOOKUP_PATH = os.path.join(THIS_DIR, "data", "subheading_names.tsv")
TEST_SET_DATA_PATH = os.path.join(THIS_DIR, "data", "test_set_data.json.gz")
TEST_SET_EXPECTED_PREDICTIONS_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Listwise22Avg_Results.json.gz")
TEST_SET_GROUND_TRUTH_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Ground_Truth.json.gz")


@pytest.mark.integration
class TestDescriptorPredictionPipeline(TestCase):

    def setUp(self):
        self.pipeline = create_async_descriptor_prediction_pipeline(DESC_NAME_LOOKUP_PATH, 
                                            DUI_LOOKUP_PATH, 
                                            "raear-cnn-endpoint-2022-v1-async", 
                                            "raear-pointwise-endpoint-2022-v2-async", 
                                            "raear-listwise-endpoint-2022-v2-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_first_five_articles(self):
        limit = 5
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
        predictions = self._predict(limit)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(gzip.open(TEST_SET_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)
        precision, recall, f1score = self._compute_metrics(ground_truth, predictions)
        
        self.assertAlmostEqual(f1score,   0.6955, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected.")
        self.assertAlmostEqual(precision, 0.7238, delta=delta, msg=f"Precision of {precision:.4f} not as expected.")
        self.assertAlmostEqual(recall,    0.6694, delta=delta, msg=f"Recall of {recall:.4f} not as expected.")
        
    def _compute_metrics(self, ground_truth, predictions):
        gt_term_name_dict = self._extract_term_names(ground_truth)
        pred_term_name_dict = self._extract_term_names(predictions)

        match_count = 0
        gt_count = 0
        pred_count = 0
        for pmid in pred_term_name_dict:
            pred_term_names = pred_term_name_dict[pmid]
            pred_count += len(pred_term_names)
            gt_term_names = gt_term_name_dict[pmid]
            gt_count += len(gt_term_names)
            matching_term_names = gt_term_names.intersection(pred_term_names)
            match_count += len(matching_term_names)

        epsilon = 1e-9
        precision = match_count / (pred_count + epsilon)
        recall = match_count / (gt_count + epsilon)
        f1score = (2*precision*recall) / (precision + recall + epsilon)
        return precision, recall, f1score

    def _extract_term_names(self, prediction_list):
        term_name_dict = {}
        for prediction in prediction_list:
            pmid = prediction["PMID"]
            term_names = { term_data["Term"] for term_data in prediction["Indexing"] }
            term_name_dict[pmid] = term_names
        return term_name_dict

    def _predict(self, limit, batch_size=512):
        test_data = self.test_set_data[:limit]
        citation_count = len(test_data)
        num_batches = int(math.ceil(citation_count / batch_size))

        predictions = []
        for idx in range(num_batches):
            batch_start = idx * batch_size
            batch_end = (idx + 1) * batch_size
            batch_inputs = test_data[batch_start:batch_end]
            batch_predictions = self.pipeline.predict(batch_inputs)
            predictions.extend(batch_predictions)
        return predictions

    def test_replace_brackets(self):
        # PMID 33998125 contains an abstract with the pattern "] [".
        self.pipeline.predict(ARTICLE_33998125)


# @pytest.mark.integration
# class TestSubheadingAttachmentPipeline(TestCase):

#     def setUp(self):
#         self.pipeline = create_subheading_attachment_pipeline(SUBHEADING_NAME_LOOKUP_PATH, 
#                                                             "raear-all-subheading-cnn-endpoint-2022-v1", 
#                                                             None,
#                                                             None,
#                                                             batch_size=128)
#         self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

#     def test_output_for_first_five_articles(self):
#         limit = 5
#         descriptor_predictions = json.load(gzip.open(TEST_SET_EXPECTED_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
#         expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
#         predictions = self._predict(descriptor_predictions, limit)
#         self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

#     def _predict(self, descriptor_predictions, limit, batch_size=512):
#         test_data = self.test_set_data[:limit]
#         citation_count = len(test_data)
#         num_batches = int(math.ceil(citation_count / batch_size))

#         predictions = []
#         for idx in range(num_batches):
#             batch_start = idx * batch_size
#             batch_end = (idx + 1) * batch_size
#             batch_inputs = test_data[batch_start:batch_end]
#             batch_descriptor_predictions = descriptor_predictions[batch_start:batch_end]
#             batch_predictions = self.pipeline.predict(batch_inputs, batch_descriptor_predictions)
#             predictions.extend(batch_predictions)
#         return predictions