from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor
from .test_data import *
from unittest import TestCase
#from unittest.mock import MagicMock, Mock


class TestCnnModelTopNPredictor(TestCase):

    def test_predict(self):
        tensorflow_predictor = None
        cnn_predictor = CnnModelTopNPredictor(tensorflow_predictor)
        top_results = cnn_predictor.predict(EXPECTED_CITATION_DATA)
        self.assertIsNotNone(top_results)