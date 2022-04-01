from .data import *
from mtix_descriptor_prediction_pipeline.predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor
from nose.plugins.attrib import attr
import random
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock


def round_top_results(top_results, ndigits):
    top_results = {q_id: {p_id: round(score, ndigits) for p_id, score in sorted(top_results[q_id].items(), key=lambda x: x[1], reverse=True) } for q_id in top_results}
    return top_results


def shuffle_top_results(top_results):
    top_results = {q_id: {p_id: score for p_id, score in random.sample(list(top_results[q_id].items()), k=len(top_results[q_id])) } for q_id in top_results}
    return top_results


@attr(test_type="unit")
class TestCnnModelTop100Predictor(TestCase):

    def test_predict(self):
        tensorflow_predictor = Mock()
        tensorflow_predictor.predict = MagicMock(
            return_value=TENSORFLOW_PREDICTOR_RESULT)
        cnn_predictor = CnnModelTop100Predictor(tensorflow_predictor)
        top_results = cnn_predictor.predict(EXPECTED_CITATION_DATA)
        top_results = round_top_results(top_results, 4)
        cnn_results = round_top_results(CNN_RESULTS, 4)
        self.assertEqual(top_results, cnn_results,
                         "top results not as expected.")
        tensorflow_predictor.predict.assert_called_once_with(
            TENSORFLOW_PREDICTOR_EXPECTED_INPUT_DATA)


@attr(test_type="unit")
class TestPointwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        
        def mock_predict(data):
            if data == HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_1:
                return HUGGINGFACE_PREDICTOR_POINTWISE_RESULTS_1
            elif data == HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_2:
                return HUGGINGFACE_PREDICTOR_POINTWISE_RESULTS_2
            else:
                raise ValueError("Huggingface predictor: unexpected input data")
        huggingface_predictor = Mock()
        huggingface_predictor.predict = Mock(side_effect=mock_predict)
        
        top_n = 5
        pointwise_predictor = PointwiseModelTopNPredictor(huggingface_predictor, DESC_NAME_LOOKUP, top_n)
        top_results = pointwise_predictor.predict(EXPECTED_CITATION_DATA, CNN_RESULTS_SHUFFLED)

        top_results = round_top_results(top_results, 6)
        expected_top_results = round_top_results(EXPECTED_POINTWISE_TOP_5_RESULTS, 6)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_predictor.predict.assert_has_calls([call(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_1), call(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_2)], any_order=True)


@attr(test_type="unit")
class TestListwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        
        huggingface_predictor = Mock()
        huggingface_predictor.predict = MagicMock(return_value=HUGGINGFACE_PREDICTOR_LISTWISE_RESULTS)
        
        top_n = 50
        listwise_predictor = ListwiseModelTopNPredictor(huggingface_predictor, DESC_NAME_LOOKUP, top_n)
        pointwise_avg_results_shuffled = shuffle_top_results(POINTWISE_AVG_RESULTS)
        top_results = listwise_predictor.predict(EXPECTED_CITATION_DATA, pointwise_avg_results_shuffled)

        top_results = round_top_results(top_results, 4)
        expected_top_results =  {q_id: {p_id: LISTWISE_RESULTS[q_id][p_id] for p_id in top_results[q_id]} for q_id in top_results}
        expected_top_results = round_top_results(expected_top_results, 4)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_predictor.predict.assert_called_once_with(HUGGINGFACE_PREDICTOR_EXPECTED_LISTWISE_INPUT_DATA)