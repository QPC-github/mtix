import gzip
import json
from mtix_descriptor_prediction_pipeline import create_descriptor_prediction_pipeline
from nose.plugins.attrib import attr
import os.path
from unittest import skip, TestCase


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESC_NAME_LOOKUP_PATH = os.path.join(THIS_DIR, "data", "main_heading_names.tsv")
DUI_LOOKUP_PATH = os.path.join(THIS_DIR, "data", "main_headings.tsv")
TEST_SET_DATA_PATH = os.path.join(THIS_DIR, "data", "test_set_data.json.gz")
TEST_SET_PREDICTIONS_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Listwise22Avg_Results.json.gz")


@attr(test_type="integration")
class TestDescriptorPredictionPipeline(TestCase):

    def setUp(self):
        self.pipeline = create_descriptor_prediction_pipeline(DESC_NAME_LOOKUP_PATH, 
                                                              DUI_LOOKUP_PATH, 
                                                              "tensorflow-inference-2022-04-01-22-15-17-484", 
                                                              "huggingface-pytorch-inference-2022-04-01-22-18-14-890", 
                                                              "huggingface-pytorch-inference-2022-04-01-22-21-50-717")

    def test_predict(self):
        limit = 2
        test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))[:limit]
        expected_predictions = json.load(gzip.open(TEST_SET_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:limit]
        
        predictions = []
        for citation_data in test_set_data:
            citation_descriptors = self.pipeline.predict([citation_data])
            predictions.extend(citation_descriptors)

        self.assertEqual(predictions, expected_predictions, "Descriptor predictions not as expected.")