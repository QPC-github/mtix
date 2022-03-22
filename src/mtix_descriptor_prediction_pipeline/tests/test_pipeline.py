import json
from mtix_descriptor_prediction_pipeline.pipeline import DescriptorPredictionPipeline, MtiJsonResultsFormatter
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor
from unittest import TestCase
#from unittest.mock import MagicMock

# TODO: Data should be included in package
TEST_SET_DATA_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_data/test_set_data.json"
TEST_SET_PREDICTIONS_PATH = "/home/raear/working_dir/mtix/scripts/create_test_set_predictions/test_set_2017-2022_Listwise22Avg_Results.json"


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