from mtix_descriptor_prediction_pipeline.predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor
import random
from .test_data import *
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock


QUERY_1 = "2017-2021|International journal of sports medicine|Second Ventilatory Threshold Assessed by Heart Rate Variability in a Multiple Shuttle Run Test.|Many studies have focused on heart rate variability in association with ventilatory thresholds. The purpose of the current study was to consider the ECG-derived respiration and the high frequency product of heart rate variability as applicable methods to assess the second ventilatory threshold (VT2). Fifteen healthy young soccer players participated in the study. Respiratory gases and ECGs were collected during an incremental laboratory test and in a multistage shuttle run test until exhaustion. VΤ2 was individually calculated using the deflection point of ventilatory equivalents. In addition, VT2 was assessed both by the deflection point of ECG-derived respiration and high frequency product. Results showed no statistically significant differences between VT2, and the threshold as determined with high frequency product and ECG-derived respiration (F(2,28)=0.83, p=0.45, η2=0.05). A significant intraclass correlation was observed for ECG-derived respiration (r=0.94) and high frequency product (r=0.95) with VT2. Similarly, Bland Altman analysis showed a considerable agreement between VT2 vs. ECG-derived respiration (mean difference of -0.06\u2009km·h-1, 95% CL: ±0.40) and VT2 vs. high frequency product (mean difference of 0.02\u2009km·h-1, 95% CL: ±0.38). This study suggests that, high frequency product and ECG-derived respiration are indeed reliable heart rate variability indices determining VT2 in a field shuttle run test."
QUERY_2 = "2017-2021|Journal of investigative medicine : the official publication of the American Federation for Clinical Research|Update on the biology and management of renal cell carcinoma.|Renal cell cancer (RCC) (epithelial carcinoma of the kidney) represents 2%-4% of newly diagnosed adult tumors. Over the past 2 decades, RCC has been better characterized clinically and molecularly. It is a heterogeneous disease, with multiple subtypes, each with characteristic histology, genetics, molecular profiles, and biologic behavior. Tremendous heterogeneity has been identified with many distinct subtypes characterized. There are clinical questions to be addressed at every stage of this disease, and new targets being identified for therapeutic development. The unique characteristics of the clinical presentations of RCC have led to both questions and opportunities for improvement in management. Advances in targeted drug development and understanding of immunologic control of RCC are leading to a number of new clinical trials and regimens for advanced disease, with the goal of achieving long-term disease-free survival, as has been achieved in a proportion of such patients historically. RCC management is a promising area of ongoing clinical investigation."


HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_1 = {
    "inputs": [
            [[ QUERY_1, "Humans"]],
            [[ QUERY_1, "Heart Rate"]],
            [[ QUERY_1, "Soccer"]],
            [[ QUERY_1, "Exercise Test"]],
            [[ QUERY_1, "Male"]],
    ],
    "parameters": {"max_length": 512,
                   "padding": "max_length",
                   "truncation": "longest_first",
                   "return_all_scores": True, },
}


HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_2 = {
    "inputs": [
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


HUGGINGFACE_PREDICTOR_POINTWISE_RESULTS_1 = [
    [{'label': 'LABEL_0', 'score': 0.0003322575648780912}, {'label': 'LABEL_1', 'score': 0.999667763710022}], 
    [{'label': 'LABEL_0', 'score': 0.000692102883476764}, {'label': 'LABEL_1', 'score': 0.9993078708648682}], 
    [{'label': 'LABEL_0', 'score': 0.024880778044462204}, {'label': 'LABEL_1', 'score': 0.9751191735267639}], 
    [{'label': 'LABEL_0', 'score': 0.05469166859984398}, {'label': 'LABEL_1', 'score': 0.9453083872795105}], 
    [{'label': 'LABEL_0', 'score': 0.010324337519705296}, {'label': 'LABEL_1', 'score': 0.9896757006645203}]] 


HUGGINGFACE_PREDICTOR_POINTWISE_RESULTS_2 = [
    [{'label': 'LABEL_0', 'score': 0.00033282057847827673}, {'label': 'LABEL_1', 'score': 0.9996671676635742}], 
    [{'label': 'LABEL_0', 'score': 0.00033223856007680297}, {'label': 'LABEL_1', 'score': 0.999667763710022}], 
    [{'label': 'LABEL_0', 'score': 0.0009819401893764734}, {'label': 'LABEL_1', 'score': 0.9990180730819702}], 
    [{'label': 'LABEL_0', 'score': 0.10123130679130554}, {'label': 'LABEL_1', 'score': 0.8987686634063721}], 
    [{'label': 'LABEL_0', 'score': 0.22837106883525848}, {'label': 'LABEL_1', 'score': 0.7716289162635803}]]


EXPECTED_POINTWISE_TOP_RESULTS = {
    "32770536": {
        "9291": 0.999667763710022,
        "8857": 0.9993078708648682,
        "15149": 0.9751192927360535,
        "7653": 0.9453081488609314,
        "10719": 0.9896757006645203,
    },
    "30455223": {
        "9291": 0.9996671676635742,
        "4978": 0.999667763710022,
        "10133": 0.9990180730819702,
        "28767": 0.8987686038017273,
        "5649": 0.7716287970542908,
    }
}

LISTWISE_PASSAGE_1 = "|Humans|Heart Rate|Soccer|Male|Electrocardiography|Exercise Test|Running|Young Adult|Anaerobic Threshold|Adolescent|Oxygen Consumption|Pulmonary Ventilation|Respiration|Athletes|Respiratory Rate|Reproducibility of Results|Pulmonary Gas Exchange|Adult|Carbon Dioxide|Physical Endurance|Cardiorespiratory Fitness|Healthy Volunteers|Athletic Performance|Fatigue|Female|Physical Exertion|Lactic Acid|Cross-Sectional Studies|Breath Tests|Oxygen|Exercise Tolerance|Heart Rate Determination|Analysis of Variance|Respiratory Function Tests|Prospective Studies|Lung|Maximal Voluntary Ventilation|Muscle Fatigue|Signal Processing, Computer-Assisted|Heart|Respiratory Mechanics|Tidal Volume|Oximetry|Physical Fitness|Spirometry|Exercise|Monitoring, Physiologic|Energy Metabolism|Respiratory Physiological Phenomena|Reference Values"
LISTWISE_PASSAGE_2 = "|Humans|Carcinoma, Renal Cell|Kidney Neoplasms|Molecular Targeted Therapy|Disease Management|Antineoplastic Agents|Clinical Trials as Topic|Disease-Free Survival|Immunotherapy|Animals|Neoplasm Staging|Biomarkers, Tumor|Prognosis|Adult|Nephrectomy|Kidney|Combined Modality Therapy|Treatment Outcome|Mutation|Signal Transduction|Neoplasm Metastasis|Disease Progression|Female|Gene Expression Regulation, Neoplastic|Male|Drug Development|Protein Kinase Inhibitors|Antineoplastic Combined Chemotherapy Protocols|Tumor Microenvironment|Genetic Predisposition to Disease|Drug Discovery|Neoplasm Grading|Immunomodulation|Disease Susceptibility|Phenylurea Compounds|Survival Analysis|Immune Checkpoint Inhibitors|Phenotype|Risk Factors|Antineoplastic Agents, Immunological|Medical Oncology|Survival Rate|Angiogenesis Inhibitors|Antibodies, Monoclonal|Immunologic Factors|Genomics|Practice Guidelines as Topic|Progression-Free Survival|Neoplasm Recurrence, Local|Biomarkers"


HUGGINGFACE_PREDICTOR_EXPECTED_LISTWISE_INPUT_DATA = {
        "inputs": [
                    [[QUERY_1, LISTWISE_PASSAGE_1]],
                    [[QUERY_2, LISTWISE_PASSAGE_2]],
                     ],
        "parameters": {},
    }


HUGGINGFACE_PREDICTOR_LISTWISE_RESULTS = [[{'index': 49, 'score': 0.008314907550811768}, {'index': 48, 'score': 0.010694682598114014}, {'index': 47, 'score': 0.0077170729637146}, {'index': 46, 'score': 0.0032181739807128906}, {'index': 45, 'score': 0.009378552436828613}, {'index': 44, 'score': 0.016565918922424316}, {'index': 43, 'score': 0.019446969032287598}, {'index': 42, 'score': 0.007498681545257568}, {'index': 41, 'score': 0.02445453405380249}, {'index': 40, 'score': 0.01658695936203003}, {'index': 39, 'score': 0.0018864870071411133}, {'index': 38, 'score': 0.012575030326843262}, {'index': 37, 'score': 0.04155451059341431}, {'index': 36, 'score': 0.020561635494232178}, {'index': 35, 'score': 0.011802196502685547}, {'index': 34, 'score': 0.02399665117263794}, {'index': 33, 'score': 0.04163938760757446}, {'index': 32, 'score': 0.010861396789550781}, {'index': 31, 'score': 0.002944469451904297}, {'index': 30, 'score': 0.024753332138061523}, {'index': 29, 'score': 0.05184274911880493}, {'index': 28, 'score': 0.05693739652633667}, {'index': 27, 'score': 0.07643085718154907}, {'index': 26, 'score': 0.036366045475006104}, {'index': 25, 'score': 0.027889370918273926}, {'index': 24, 'score': 0.10209900140762329}, {'index': 23, 'score': 0.08230847120285034}, {'index': 22, 'score': 0.029399514198303223}, {'index': 21, 'score': 0.01474231481552124}, {'index': 20, 'score': 0.06415504217147827}, {'index': 19, 'score': 0.16788339614868164}, {'index': 18, 'score': 0.19816654920578003}, {'index': 17, 'score': 0.2837294340133667}, {'index': 16, 'score': 0.08189046382904053}, {'index': 15, 'score': 0.13493561744689941}, {'index': 14, 'score': 0.1243436336517334}, {'index': 13, 'score': 0.05410969257354736}, {'index': 12, 'score': 0.5234928727149963}, {'index': 11, 'score': 0.5007476210594177}, {'index': 10, 'score': 0.6322138905525208}, {'index': 9, 'score': 0.5184025764465332}, {'index': 8, 'score': 0.48771417140960693}, {'index': 7, 'score': 0.8378668427467346}, {'index': 6, 'score': 0.8620148301124573}, {'index': 5, 'score': 0.9710224866867065}, {'index': 4, 'score': 0.9671317338943481}, {'index': 3, 'score': 0.8103287220001221}, {'index': 2, 'score': 0.8039405941963196}, {'index': 1, 'score': 0.9983595013618469}, {'index': 0, 'score': 0.9997273683547974}], [{'index': 49, 'score': 0.008774638175964355}, {'index': 48, 'score': 0.013961374759674072}, {'index': 47, 'score': 0.005237460136413574}, {'index': 46, 'score': 0.012418806552886963}, {'index': 45, 'score': 0.0184171199798584}, {'index': 44, 'score': 0.012364447116851807}, {'index': 43, 'score': 0.0068038105964660645}, {'index': 42, 'score': 0.010458767414093018}, {'index': 41, 'score': 0.004331231117248535}, {'index': 40, 'score': 0.007567703723907471}, {'index': 39, 'score': 0.015400826930999756}, {'index': 38, 'score': 0.03832584619522095}, {'index': 37, 'score': 0.03976106643676758}, {'index': 36, 'score': 0.010516107082366943}, {'index': 35, 'score': 0.009590327739715576}, {'index': 34, 'score': 0.003970503807067871}, {'index': 33, 'score': 0.012640535831451416}, {'index': 32, 'score': 0.011380136013031006}, {'index': 31, 'score': 0.02446889877319336}, {'index': 30, 'score': 0.006635487079620361}, {'index': 29, 'score': 0.02613067626953125}, {'index': 28, 'score': 0.02938741445541382}, {'index': 27, 'score': 0.02096712589263916}, {'index': 26, 'score': 0.010129809379577637}, {'index': 25, 'score': 0.021213114261627197}, {'index': 24, 'score': 0.01467907428741455}, {'index': 23, 'score': 0.01876533031463623}, {'index': 22, 'score': 0.016844749450683594}, {'index': 21, 'score': 0.033575236797332764}, {'index': 20, 'score': 0.10350322723388672}, {'index': 19, 'score': 0.04089772701263428}, {'index': 18, 'score': 0.053028881549835205}, {'index': 17, 'score': 0.057190537452697754}, {'index': 16, 'score': 0.05736345052719116}, {'index': 15, 'score': 0.011299669742584229}, {'index': 14, 'score': 0.01954352855682373}, {'index': 13, 'score': 0.04465007781982422}, {'index': 12, 'score': 0.08914214372634888}, {'index': 11, 'score': 0.12910079956054688}, {'index': 10, 'score': 0.1276611089706421}, {'index': 9, 'score': 0.12334918975830078}, {'index': 8, 'score': 0.07487016916275024}, {'index': 7, 'score': 0.13302522897720337}, {'index': 6, 'score': 0.261746883392334}, {'index': 5, 'score': 0.10084354877471924}, {'index': 4, 'score': 0.2031126618385315}, {'index': 3, 'score': 0.315798282623291}, {'index': 2, 'score': 0.8500163555145264}, {'index': 1, 'score': 0.9988843202590942}, {'index': 0, 'score': 0.9994417428970337}]]


def round_top_results(top_results, ndigits):
    top_results = {q_id: {p_id: round(score, ndigits) for p_id, score in sorted(top_results[q_id].items(), key=lambda x: x[1], reverse=True) } for q_id in top_results}
    return top_results


def shuffle_top_results(top_results):
    top_results = {q_id: {p_id: score for p_id, score in random.sample(list(top_results[q_id].items()), k=len(top_results[q_id])) } for q_id in top_results}
    return top_results


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
        expected_top_results = round_top_results(EXPECTED_POINTWISE_TOP_RESULTS, 6)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_predictor.predict.assert_has_calls([call(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_1), call(HUGGINGFACE_PREDICTOR_EXPECTED_POINTWISE_INPUT_DATA_2)], any_order=True)


class TestListwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        
        huggingface_predictor = Mock()
        huggingface_predictor.predict = MagicMock(return_value=HUGGINGFACE_PREDICTOR_LISTWISE_RESULTS)
        
        top_n = 5
        listwise_predictor = ListwiseModelTopNPredictor(huggingface_predictor, DESC_NAME_LOOKUP, top_n)
        pointwise_avg_results_shuffled = shuffle_top_results(POINTWISE_AVG_RESULTS)
        top_results = listwise_predictor.predict(EXPECTED_CITATION_DATA, pointwise_avg_results_shuffled)

        top_results = round_top_results(top_results, 6)
        expected_top_results =  {q_id: {p_id: LISTWISE_RESULTS[q_id][p_id] for p_id in top_results[q_id]} for q_id in top_results}
        expected_top_results = round_top_results(expected_top_results, 6)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_predictor.predict.assert_called_once_with(HUGGINGFACE_PREDICTOR_EXPECTED_LISTWISE_INPUT_DATA)