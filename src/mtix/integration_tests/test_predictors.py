from .data import *
import gzip
import json
import math
from mtix.sagemaker_factory import create_subheading_predictor
import os.path
import pytest
from unittest import skip, TestCase


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
SUBHEADING_NAME_LOOKUP_PATH =                   os.path.join(DATA_DIR, "subheading_names.tsv")
TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH =         os.path.join(DATA_DIR, "test_set_2017-2022_Descriptor_Ground_Truth.json.gz")
TEST_SET_EXPECTED_SUBHEADING_PREDICTIONS_PATH = os.path.join(DATA_DIR, "test_set_2017-2022_Subheading_Predictions.json.gz")
TEST_SET_SUBHEADING_GROUND_TRUTH_PATH =         os.path.join(DATA_DIR, "test_set_2017-2022_Subheading_Ground_Truth.json.gz")


@pytest.mark.integration
class TestSubheadingPredictor(TestCase):

    def setUp(self):
        self.pipeline = create_subheading_predictor(SUBHEADING_NAME_LOOKUP_PATH, 
                                                    "raear-all-subheading-cnn-endpoint-2022-v1-async", 
                                                    "ncbi-aws-pmdm-ingest",
                                                    "async_inference",
                                                     batch_size=128)
        self.descriptor_predictions = json.load(gzip.open(TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))

    def test_output_for_first_five_articles(self):
        limit = 5
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_SUBHEADING_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
        predictions = self._predict(limit)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def _predict(self, limit, batch_size=512):
        test_data = self.descriptor_predictions[:limit]
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