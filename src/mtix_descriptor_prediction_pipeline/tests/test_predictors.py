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
                    [["|" + QUERY_1, LISTWISE_PASSAGE_1]],
                    [["|" + QUERY_2, LISTWISE_PASSAGE_2]],
                     ],
        "parameters": {},
    }


HUGGINGFACE_PREDICTOR_LISTWISE_RESULTS = [[{'index': 49, 'score': 0.007938742637634277}, {'index': 48, 'score': 0.010896086692810059}, {'index': 47, 'score': 0.006842315196990967}, {'index': 46, 'score': 0.0037491321563720703}, {'index': 45, 'score': 0.01112520694732666}, {'index': 44, 'score': 0.014872968196868896}, {'index': 43, 'score': 0.021402597427368164}, {'index': 42, 'score': 0.006919980049133301}, {'index': 41, 'score': 0.019832611083984375}, {'index': 40, 'score': 0.014018476009368896}, {'index': 39, 'score': 0.0018600225448608398}, {'index': 38, 'score': 0.012973129749298096}, {'index': 37, 'score': 0.037304580211639404}, {'index': 36, 'score': 0.018665552139282227}, {'index': 35, 'score': 0.011301875114440918}, {'index': 34, 'score': 0.02417898178100586}, {'index': 33, 'score': 0.04350954294204712}, {'index': 32, 'score': 0.011007189750671387}, {'index': 31, 'score': 0.0035671591758728027}, {'index': 30, 'score': 0.027734100818634033}, {'index': 29, 'score': 0.05508875846862793}, {'index': 28, 'score': 0.051951587200164795}, {'index': 27, 'score': 0.07606261968612671}, {'index': 26, 'score': 0.035243332386016846}, {'index': 25, 'score': 0.022531330585479736}, {'index': 24, 'score': 0.09748035669326782}, {'index': 23, 'score': 0.07878339290618896}, {'index': 22, 'score': 0.02871263027191162}, {'index': 21, 'score': 0.017262935638427734}, {'index': 20, 'score': 0.06295996904373169}, {'index': 19, 'score': 0.16926616430282593}, {'index': 18, 'score': 0.17700445652008057}, {'index': 17, 'score': 0.27509385347366333}, {'index': 16, 'score': 0.07260715961456299}, {'index': 15, 'score': 0.14152204990386963}, {'index': 14, 'score': 0.1057196855545044}, {'index': 13, 'score': 0.07359951734542847}, {'index': 12, 'score': 0.5544146299362183}, {'index': 11, 'score': 0.4863630533218384}, {'index': 10, 'score': 0.6221471428871155}, {'index': 9, 'score': 0.5496633052825928}, {'index': 8, 'score': 0.5114960074424744}, {'index': 7, 'score': 0.8315696716308594}, {'index': 6, 'score': 0.852420449256897}, {'index': 5, 'score': 0.9706422090530396}, {'index': 4, 'score': 0.9643720388412476}, {'index': 3, 'score': 0.8125095963478088}, {'index': 2, 'score': 0.7955935001373291}, {'index': 1, 'score': 0.9982964396476746}, {'index': 0, 'score': 0.9997355341911316}], [{'index': 49, 'score': 0.00880122184753418}, {'index': 48, 'score': 0.013546645641326904}, {'index': 47, 'score': 0.005581796169281006}, {'index': 46, 'score': 0.01275014877319336}, {'index': 45, 'score': 0.01950240135192871}, {'index': 44, 'score': 0.011992573738098145}, {'index': 43, 'score': 0.006782352924346924}, {'index': 42, 'score': 0.010310649871826172}, {'index': 41, 'score': 0.004575967788696289}, {'index': 40, 'score': 0.007065236568450928}, {'index': 39, 'score': 0.013955831527709961}, {'index': 38, 'score': 0.035296082496643066}, {'index': 37, 'score': 0.038269758224487305}, {'index': 36, 'score': 0.010201454162597656}, {'index': 35, 'score': 0.009143590927124023}, {'index': 34, 'score': 0.003990054130554199}, {'index': 33, 'score': 0.012099087238311768}, {'index': 32, 'score': 0.011150598526000977}, {'index': 31, 'score': 0.0269661545753479}, {'index': 30, 'score': 0.005804121494293213}, {'index': 29, 'score': 0.02430778741836548}, {'index': 28, 'score': 0.02986544370651245}, {'index': 27, 'score': 0.018231868743896484}, {'index': 26, 'score': 0.009944558143615723}, {'index': 25, 'score': 0.021324455738067627}, {'index': 24, 'score': 0.014104783535003662}, {'index': 23, 'score': 0.016591429710388184}, {'index': 22, 'score': 0.01630711555480957}, {'index': 21, 'score': 0.03344601392745972}, {'index': 20, 'score': 0.10456740856170654}, {'index': 19, 'score': 0.039521872997283936}, {'index': 18, 'score': 0.05310338735580444}, {'index': 17, 'score': 0.04900604486465454}, {'index': 16, 'score': 0.05385619401931763}, {'index': 15, 'score': 0.011100411415100098}, {'index': 14, 'score': 0.019696176052093506}, {'index': 13, 'score': 0.04421919584274292}, {'index': 12, 'score': 0.08873981237411499}, {'index': 11, 'score': 0.13066619634628296}, {'index': 10, 'score': 0.1370680332183838}, {'index': 9, 'score': 0.11485886573791504}, {'index': 8, 'score': 0.06807821989059448}, {'index': 7, 'score': 0.12978792190551758}, {'index': 6, 'score': 0.26191896200180054}, {'index': 5, 'score': 0.09714454412460327}, {'index': 4, 'score': 0.19595825672149658}, {'index': 3, 'score': 0.3219943046569824}, {'index': 2, 'score': 0.8412795662879944}, {'index': 1, 'score': 0.9989158511161804}, {'index': 0, 'score': 0.9994300007820129}]]


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