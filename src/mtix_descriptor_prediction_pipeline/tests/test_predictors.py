from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor
from .test_data import *
from unittest import TestCase
from unittest.mock import MagicMock, Mock

#TODO: does it make senese to continue to use pytrec_eval? with string keys?

QUERY_1 = "2017-2021|International journal of sports medicine|Second Ventilatory Threshold Assessed by Heart Rate Variability in a Multiple Shuttle Run Test.|Many studies have focused on heart rate variability in association with ventilatory thresholds. The purpose of the current study was to consider the ECG-derived respiration and the high frequency product of heart rate variability as applicable methods to assess the second ventilatory threshold (VT2). Fifteen healthy young soccer players participated in the study. Respiratory gases and ECGs were collected during an incremental laboratory test and in a multistage shuttle run test until exhaustion. VΤ2 was individually calculated using the deflection point of ventilatory equivalents. In addition, VT2 was assessed both by the deflection point of ECG-derived respiration and high frequency product. Results showed no statistically significant differences between VT2, and the threshold as determined with high frequency product and ECG-derived respiration (F(2,28)=0.83, p=0.45, η2=0.05). A significant intraclass correlation was observed for ECG-derived respiration (r=0.94) and high frequency product (r=0.95) with VT2. Similarly, Bland Altman analysis showed a considerable agreement between VT2 vs. ECG-derived respiration (mean difference of -0.06\u2009km·h-1, 95% CL: ±0.40) and VT2 vs. high frequency product (mean difference of 0.02\u2009km·h-1, 95% CL: ±0.38). This study suggests that, high frequency product and ECG-derived respiration are indeed reliable heart rate variability indices determining VT2 in a field shuttle run test."
QUERY_2 = "2017-2021|Journal of investigative medicine : the official publication of the American Federation for Clinical Research|Update on the biology and management of renal cell carcinoma.|Renal cell cancer (RCC) (epithelial carcinoma of the kidney) represents 2%-4% of newly diagnosed adult tumors. Over the past 2 decades, RCC has been better characterized clinically and molecularly. It is a heterogeneous disease, with multiple subtypes, each with characteristic histology, genetics, molecular profiles, and biologic behavior. Tremendous heterogeneity has been identified with many distinct subtypes characterized. There are clinical questions to be addressed at every stage of this disease, and new targets being identified for therapeutic development. The unique characteristics of the clinical presentations of RCC have led to both questions and opportunities for improvement in management. Advances in targeted drug development and understanding of immunologic control of RCC are leading to a number of new clinical trials and regimens for advanced disease, with the goal of achieving long-term disease-free survival, as has been achieved in a proportion of such patients historically. RCC management is a promising area of ongoing clinical investigation."
                
HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA = {
    "inputs": [
            [[ QUERY_1, "Humans"]],
            [[ QUERY_1, "Heart Rate"]],
            [[ QUERY_1, "Soccer"]],
            [[ QUERY_1, "Exercise Test"]],
            [[ QUERY_1, "Male"]],
            [[ QUERY_2, "Humans"]],
            [[ QUERY_2, "Carcinoma, Renal Cell"]],
            [[ QUERY_2, "Kidney Neoplasms"]],
            [[ QUERY_2, "Molecular Targeted Therapy"]],
            [[ QUERY_2, "Clinical Trials as Topic"]],
    ],
    "parameters": {"max_length": 512,
                   "padding": "max_length",
                   "truncation": "longest_first",
                   "return_all_scores": True, },
}


HUGGINGFACE_PREDICTOR_POINTWISE_RESULT = [
    [{'label': 'LABEL_0', 'score': 0.0003322575648780912}, {'label': 'LABEL_1', 'score': 0.999667763710022}], 
    [{'label': 'LABEL_0', 'score': 0.000692102883476764}, {'label': 'LABEL_1', 'score': 0.9993078708648682}], 
    [{'label': 'LABEL_0', 'score': 0.024880778044462204}, {'label': 'LABEL_1', 'score': 0.9751191735267639}], 
    [{'label': 'LABEL_0', 'score': 0.05469166859984398}, {'label': 'LABEL_1', 'score': 0.9453083872795105}], 
    [{'label': 'LABEL_0', 'score': 0.010324337519705296}, {'label': 'LABEL_1', 'score': 0.9896757006645203}], 
    [{'label': 'LABEL_0', 'score': 0.00033282057847827673}, {'label': 'LABEL_1', 'score': 0.9996671676635742}], 
    [{'label': 'LABEL_0', 'score': 0.00033223856007680297}, {'label': 'LABEL_1', 'score': 0.999667763710022}], 
    [{'label': 'LABEL_0', 'score': 0.0009819401893764734}, {'label': 'LABEL_1', 'score': 0.9990180730819702}], 
    [{'label': 'LABEL_0', 'score': 0.10123130679130554}, {'label': 'LABEL_1', 'score': 0.8987686634063721}], 
    [{'label': 'LABEL_0', 'score': 0.22837106883525848}, {'label': 'LABEL_1', 'score': 0.7716289162635803}]]


def round_top_results(top_results, ndigits, top_n=10000000):
    top_results = {q_id: {p_id: round(score, ndigits) for p_id, score in sorted(top_results[q_id].items(), key=lambda x: x[1], reverse=True)[:top_n] } for q_id in top_results}
    return top_results


class TestCnnModelTopNPredictor(TestCase):

    def test_predict(self):
        tensorflow_predictor = Mock()
        tensorflow_predictor.predict = MagicMock(
            return_value=TENSORFLOW_PREDICTOR_RESULT)
        cnn_predictor = CnnModelTopNPredictor(tensorflow_predictor)
        top_results = cnn_predictor.predict(EXPECTED_CITATION_DATA)
        top_results = round_top_results(top_results, 4)
        cnn_results = round_top_results(CNN_RESULTS, 4)
        self.assertEqual(top_results, cnn_results,
                         "top results not as expected.")
        tensorflow_predictor.predict.assert_called_once_with(
            TENSORFLOW_PREDICTOR_EXPECTED_INPUT_DATA)


class TestPointwiseModelTopNPredictor(TestCase):

    # TODO: need to test that CNN top results are sorted
    def test_predict(self):
        huggingface_predictor = Mock()
        huggingface_predictor.predict = MagicMock(
            return_value=HUGGINGFACE_PREDICTOR_POINTWISE_RESULT)
        top_n = 5
        pointwise_predictor = PointwiseModelTopNPredictor(
            huggingface_predictor, DESC_NAME_LOOKUP, top_n)
        top_results = pointwise_predictor.predict(
            EXPECTED_CITATION_DATA, CNN_RESULTS)
        top_results = round_top_results(top_results, 6)
        expected_top_results =  {q_id: {p_id: POINTWISE_RESULTS[q_id][p_id] for p_id in top_results[q_id]} for q_id in top_results}
        expected_top_results = round_top_results(expected_top_results, 6)
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        huggingface_predictor.predict.assert_called_once_with(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA)