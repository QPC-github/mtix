from .pipeline import CitationDataSanitizer, DescriptorPredictionPipeline, MedlineDateParser, MtiJsonResultsFormatter, PubMedXmlInputDataParser
from .predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.tensorflow import TensorFlowPredictor
from .utils import create_lookup


def create_descriptor_prediction_pipeline(desc_name_lookup_path, dui_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name):
    max_year = 2021
    listwise_top_n = 50
    pointwise_top_n = 100
    threshold = 0.475

    medline_date_parser = MedlineDateParser()
    input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
    
    sanitizer = CitationDataSanitizer(max_year)

    tensorflow_predictor = TensorFlowPredictor(cnn_endpoint_name)
    cnn_model_top_100_predictor = CnnModelTop100Predictor(tensorflow_predictor)

    desc_name_lookup = create_lookup(desc_name_lookup_path)
    pointwise_hugginface_predictor = HuggingFacePredictor(pointwise_endpoint_name)
    pointwise_model_top100_predictor = PointwiseModelTopNPredictor(pointwise_hugginface_predictor, desc_name_lookup, pointwise_top_n)

    listwise_hugginface_predictor = HuggingFacePredictor(listwise_endpoint_name)
    listwise_model_top50_predictor = ListwiseModelTopNPredictor(listwise_hugginface_predictor, desc_name_lookup, listwise_top_n)

    dui_lookup = create_lookup(dui_lookup_path)
    results_formatter = MtiJsonResultsFormatter(desc_name_lookup, dui_lookup, threshold)

    pipeline = DescriptorPredictionPipeline(input_data_parser, sanitizer, cnn_model_top_100_predictor, pointwise_model_top100_predictor, listwise_model_top50_predictor, results_formatter)
    return pipeline