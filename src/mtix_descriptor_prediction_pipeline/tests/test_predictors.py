from mtix_descriptor_prediction_pipeline.predictors import CnnModelTopNPredictor, PointwiseModelTopNPredictor
from .test_data import *
from unittest import TestCase
from unittest.mock import MagicMock, Mock

#TODO: does it make senese to continue to use pytrec_eval? with string keys?

HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA = {
    "inputs": [
        [["2017-2021|Clinical & translational oncology : official publication of the Federation of Spanish Oncology Societies and of the National Cancer Institute of Mexico|Lysophosphatidic acid receptor 6 regulated by miR-27a-3p attenuates tumor proliferation in breast cancer.|PURPOSE: Lysophosphatidic acid (LPA) is a bioactive molecule which participates in many physical and pathological processes. Although LPA receptor 6 (LPAR6), the last identified LPA receptor, has been reported to have diverse effects in multiple cancers, including breast cancer, its effects and functioning mechanisms are not fully known.METHODS: Multiple public databases were used to investigate the mRNA expression of LPAR6, its prognostic value, and potential mechanisms in breast cancer. Western blotting was performed to validate the differential expression of LPAR6 in breast cancer tissues and their adjacent tissues. Furthermore, in vitro experiments were used to explore the effects of LPAR6 on breast cancer. Additionally, TargetScan and miRWalk were used to identify potential upstream regulating miRNAs and validated the relationship between miR-27a-3p and LPAR6 via real-time polymerase chain reaction and an in vitro rescue assay.RESULTS: LPAR6 was significantly downregulated in breast cancer at transcriptional and translational levels. Decreased LPAR6 expression in breast cancer is significantly correlated with poor overall survival, disease-free survival, and distal metastasis-free survival, particularly for hormone receptor-positive patients, regardless of lymph node metastatic status. In vitro gain and loss-of-function assays indicated that LPAR6 attenuated breast cancer cell proliferation. The analyses of TCGA and METABRIC datasets revealed that LPAR6 may regulate the cell cycle signal pathway. Furthermore, the expression of LPAR6 could be positively regulated by miR-27a-3p. The knockdown of miR-27a-3p increased cell proliferation, and ectopic expression of LPAR6 could partly rescue this phenotype.CONCLUSION: LPAR6 acts as a tumor suppressor in breast cancer and is positively regulated by miR-27a-3p.", "Humans"]],
        [["2017-2021|Clinical & translational oncology : official publication of the Federation of Spanish Oncology Societies and of the National Cancer Institute of Mexico|Lysophosphatidic acid receptor 6 regulated by miR-27a-3p attenuates tumor proliferation in breast cancer.|PURPOSE: Lysophosphatidic acid (LPA) is a bioactive molecule which participates in many physical and pathological processes. Although LPA receptor 6 (LPAR6), the last identified LPA receptor, has been reported to have diverse effects in multiple cancers, including breast cancer, its effects and functioning mechanisms are not fully known.METHODS: Multiple public databases were used to investigate the mRNA expression of LPAR6, its prognostic value, and potential mechanisms in breast cancer. Western blotting was performed to validate the differential expression of LPAR6 in breast cancer tissues and their adjacent tissues. Furthermore, in vitro experiments were used to explore the effects of LPAR6 on breast cancer. Additionally, TargetScan and miRWalk were used to identify potential upstream regulating miRNAs and validated the relationship between miR-27a-3p and LPAR6 via real-time polymerase chain reaction and an in vitro rescue assay.RESULTS: LPAR6 was significantly downregulated in breast cancer at transcriptional and translational levels. Decreased LPAR6 expression in breast cancer is significantly correlated with poor overall survival, disease-free survival, and distal metastasis-free survival, particularly for hormone receptor-positive patients, regardless of lymph node metastatic status. In vitro gain and loss-of-function assays indicated that LPAR6 attenuated breast cancer cell proliferation. The analyses of TCGA and METABRIC datasets revealed that LPAR6 may regulate the cell cycle signal pathway. Furthermore, the expression of LPAR6 could be positively regulated by miR-27a-3p. The knockdown of miR-27a-3p increased cell proliferation, and ectopic expression of LPAR6 could partly rescue this phenotype.CONCLUSION: LPAR6 acts as a tumor suppressor in breast cancer and is positively regulated by miR-27a-3p.", "Guinea Pigs"]],
    ],
    "parameters": {"max_length": 512,
                   "padding": "max_length",
                   "truncation": "longest_first",
                   "return_all_scores": True, },
}
HUGGINGFACE_PREDICTOR_POINTWISE_RESULT = [[{'label': 'LABEL_0', 'score': 0.00033122117747552693}, {'label': 'LABEL_1', 'score': 0.9996688365936279}], [{'label': 'LABEL_0', 'score': 0.9942746758460999}, {'label': 'LABEL_1', 'score': 0.005725273862481117}]]


def round_top_results(top_results, ndigits):
    top_results = { q_id: { p_id: round(top_results[q_id][p_id], ndigits) for p_id in top_results[q_id]} for q_id in top_results}
    return top_results


class TestCnnModelTopNPredictor(TestCase):

    def test_predict(self):
        tensorflow_predictor = Mock()
        tensorflow_predictor.predict = MagicMock(return_value=TENSORFLOW_PREDICTOR_RESULT)
        cnn_predictor = CnnModelTopNPredictor(tensorflow_predictor)
        top_results = cnn_predictor.predict(EXPECTED_CITATION_DATA)
        top_results = round_top_results(top_results, 4)
        cnn_results = round_top_results(CNN_RESULTS, 4)
        self.assertEqual(top_results, cnn_results, "top results not as expected.")
        tensorflow_predictor.predict.assert_called_once_with(TENSORFLOW_PREDICTOR_EXPECTED_INPUT_DATA)


class TestPointwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        huggingface_predictor = Mock()
        huggingface_predictor.predict = MagicMock(return_value=HUGGINGFACE_PREDICTOR_POINTWISE_RESULT)
        pointwise_predictor = PointwiseModelTopNPredictor(huggingface_predictor, DESC_NAME_LOOKUP, 10)
        top_results = pointwise_predictor.predict(EXPECTED_CITATION_DATA, CNN_RESULTS)
        top_results = round_top_results(top_results, 4)
        pointwise_results = round_top_results(POINTWISE_RESULTS, 4)
        self.assertEqual(top_results, pointwise_results, "top results not as expected.")
        huggingface_predictor.predict.assert_called_once_with(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA)