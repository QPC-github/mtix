from .endpoints import HuggingFaceAsyncEndpoint, HuggingFaceRealTimeEndpoint, TensorflowAsyncEndpoint, TensorflowRealTimeEndpoint
from .pipeline import CitationDataSanitizer, DescriptorPredictionPipeline, MedlineDateParser, MtiJsonResultsFormatter, PubMedXmlInputDataParser
from .predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.tensorflow import TensorFlowPredictor
from .utils import create_lookup


def create_descriptor_prediction_pipeline(desc_name_lookup_path, dui_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, bucket_name, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128):
    max_year = 2021
    listwise_top_n = 50
    pointwise_top_n = 100
    threshold = 0.475

    medline_date_parser = MedlineDateParser()
    input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
    
    sanitizer = CitationDataSanitizer(max_year)

    sagemaker_tf_endpoint = TensorFlowPredictor(cnn_endpoint_name)
    #tensorflow_endpoint = TensorflowRealTimeEndpoint(sagemaker_tf_endpoint, batch_size=cnn_batch_size)
    sagemaker_async_tf_endpoint = AsyncPredictor(sagemaker_tf_endpoint)
    tensorflow_endpoint = TensorflowAsyncEndpoint(sagemaker_async_tf_endpoint, "cnn_endpoint", bucket_name, batch_size=cnn_batch_size)
    cnn_model_top_100_predictor = CnnModelTop100Predictor(tensorflow_endpoint)

    desc_name_lookup = create_lookup(desc_name_lookup_path)
    pointwise_sagemaker_hf_endpoint = HuggingFacePredictor(pointwise_endpoint_name)
    #pointwise_hf_endpoint = HuggingFaceRealTimeEndpoint(pointwise_sagemaker_hf_endpoint, batch_size=pointwise_batch_size)
    pointwise_sagemaker_async_hf_endpoint = AsyncPredictor(pointwise_sagemaker_hf_endpoint)
    pointwise_hf_endpoint = HuggingFaceAsyncEndpoint(pointwise_sagemaker_async_hf_endpoint, "pointwise_endpoint", bucket_name, batch_size=pointwise_batch_size)
    pointwise_model_top100_predictor = PointwiseModelTopNPredictor(pointwise_hf_endpoint, desc_name_lookup, pointwise_top_n)

    listwise_sagemaker_hf_endpoint = HuggingFacePredictor(listwise_endpoint_name)
    #listwise_hf_endpoint = HuggingFaceRealTimeEndpoint(listwise_sagemaker_hf_endpoint, batch_size=listwise_batch_size)
    listwise_sagemaker_async_hf_endpoint = AsyncPredictor(listwise_sagemaker_hf_endpoint)
    listwise_hf_endpoint = HuggingFaceAsyncEndpoint(listwise_sagemaker_async_hf_endpoint, "listwise_endpoint", bucket_name, batch_size=listwise_batch_size)
    listwise_model_top50_predictor = ListwiseModelTopNPredictor(listwise_hf_endpoint, desc_name_lookup, listwise_top_n)

    dui_lookup = create_lookup(dui_lookup_path)
    results_formatter = MtiJsonResultsFormatter(desc_name_lookup, dui_lookup, threshold)

    pipeline = DescriptorPredictionPipeline(input_data_parser, sanitizer, cnn_model_top_100_predictor, pointwise_model_top100_predictor, listwise_model_top50_predictor, results_formatter)
    return pipeline