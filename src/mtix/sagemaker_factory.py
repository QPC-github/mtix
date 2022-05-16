from .endpoints import HuggingFaceAsyncEndpoint, HuggingFaceRealTimeEndpoint, TensorflowAsyncEndpoint, TensorflowRealTimeEndpoint
from .pipeline import CitationDataSanitizer, DescriptorPredictionPipeline, MedlineDateParser, MtiJsonResultsFormatter, PubMedXmlInputDataParser
from .predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.tensorflow import TensorFlowPredictor
from .utils import create_lookup


def create_descriptor_prediction_pipeline(desc_name_lookup_path, dui_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128):
    concurrent_batches=100
    max_year = 2021
    listwise_top_n = 50
    pointwise_top_n = 100
    threshold = 0.475

    medline_date_parser = MedlineDateParser()
    input_data_parser = PubMedXmlInputDataParser(medline_date_parser)
    
    sanitizer = CitationDataSanitizer(max_year)

    sagemaker_cnn_endpoint = TensorFlowPredictor(cnn_endpoint_name)
   
    sagemaker_async_cnn_endpoint = AsyncPredictor(sagemaker_cnn_endpoint)
    async_cnn_endpoint = TensorflowAsyncEndpoint(sagemaker_async_cnn_endpoint, "cnn_endpoint", async_bucket_name, async_prefix, cnn_batch_size, wait_delay=1, wait_max_attempts=300)
    cnn_endpoint = TensorflowRealTimeEndpoint(async_cnn_endpoint, batch_size=concurrent_batches*cnn_batch_size)
    cnn_model_top_100_predictor = CnnModelTop100Predictor(cnn_endpoint)

    desc_name_lookup = create_lookup(desc_name_lookup_path)
    sagemaker_pointwise_endpoint = HuggingFacePredictor(pointwise_endpoint_name)
    sagemaker_async_pointwise_endpoint = AsyncPredictor(sagemaker_pointwise_endpoint)
    async_pointwise_endpoint = HuggingFaceAsyncEndpoint(sagemaker_async_pointwise_endpoint, "pointwise_endpoint", async_bucket_name, async_prefix, pointwise_batch_size, wait_delay=1, wait_max_attempts=600)
    pointwise_endpoint = HuggingFaceRealTimeEndpoint(async_pointwise_endpoint, batch_size=concurrent_batches*pointwise_batch_size)
    pointwise_model_top100_predictor = PointwiseModelTopNPredictor(pointwise_endpoint, desc_name_lookup, pointwise_top_n)

    sagemaker_listwise_endpoint = HuggingFacePredictor(listwise_endpoint_name)

    sagemaker_async_listwise_endpoint = AsyncPredictor(sagemaker_listwise_endpoint)
    async_listwise_endpoint = HuggingFaceAsyncEndpoint(sagemaker_async_listwise_endpoint, "listwise_endpoint", async_bucket_name, async_prefix, listwise_batch_size, wait_delay=1, wait_max_attempts=300)
    listwise_endpoint = HuggingFaceRealTimeEndpoint(async_listwise_endpoint, batch_size=concurrent_batches*listwise_batch_size)
    listwise_model_top50_predictor = ListwiseModelTopNPredictor(listwise_endpoint, desc_name_lookup, listwise_top_n)

    dui_lookup = create_lookup(dui_lookup_path)
    results_formatter = MtiJsonResultsFormatter(desc_name_lookup, dui_lookup, threshold)

    pipeline = DescriptorPredictionPipeline(input_data_parser, sanitizer, cnn_model_top_100_predictor, pointwise_model_top100_predictor, listwise_model_top50_predictor, results_formatter)
    return pipeline