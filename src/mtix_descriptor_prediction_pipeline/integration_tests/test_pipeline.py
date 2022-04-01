from mtix_descriptor_prediction_pipeline.pipeline import DescriptorPredictionPipeline
from nose.plugins.attrib import attr
import os.path
from unittest import skip, TestCase


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DATA_PATH = os.path.join(THIS_DIR, "data", "test_set_data.json.gz")
TEST_SET_PREDICTIONS_PATH = os.path.join(THIS_DIR, "data", "test_set_2017-2022_Listwise22Avg_Results.json.gz")


@attr(test_type="integration")
class TestDescriptorPredictionPipeline(TestCase):

    def setUp(self):
        pass

    def test_predict(self):
        self.fail()