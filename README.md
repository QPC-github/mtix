# MTIX Descriptor Prediction Pipeline

## Installation

1. Create a Python 3.9 virtual environment. 
2. Install the package:

```
pip install .
```
3. Upload the CNN, Pointwise, and Listwise model files to an AWS S3 bucket.
4. Deploy the SageMaker endpoints using the script provided in the git repository (./scripts/create_sagemaker_endpoints.py). Deployment settings (e.g. S3 bucket name) can be modified at the top of the script.
5. Create the following S3 directories for async prediction temporary files:<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/outputs/<br>

## Test
Automated unit and integration tests can be run using pytest. To run the integration test you may need to update the SageMaker endpoint names in the integration test code. The integration tests check predictions for 40k citations, and they may therefore take a long time to run.
```
pytest -m unit
pytest -m integration
```

## Usage

The pipeline is constructed with the following input parameters:

1. Path to the Descriptor name lookup file. This file maps internal Descriptor ids to Descriptor names.
2. Path to Desciptor unique identifier file. This file maps internal Descriptor ids to NLM DUIs.
3. Sagemaker endpoint name for CNN model.
4. Sagemaker endpoint name for Pointwise model.
5. Sagemaker endpoint name for Listwise model.
6. The s3 bucket name (for async prediction temporary data).
7. The s3 prefix (for async prediction temporary data).
8. The cnn model batch size.
9. The pointwise model batch size.
10. The listwise model batch size.

Example usage for async endpoints:

```
from mtix import create_async_pipeline

pipeline = create_async_pipeline("path/to/main_heading_names.tsv", 
                                "path/to/main_heading.tsv, 
                                "dev-cnn-model-2022-v1-async", 
                                "dev-pointwise-model-2022-v2-async", 
                                "dev-listwise-model-2022-v2-async",
                                "ncbi-aws-pmdm-ingest",
                                "async_inference",
                                cnn_batch_size=128,
                                pointwise_batch_size=128,
                                listwise_batch_size=128)

desc_predictions = pipeline.predict(input_data)
```

Example usage for real-time endpoints:

```
from mtix import create_real_time_pipeline

pipeline = create_real_time_pipeline("path/to/main_heading_names.tsv", 
                                "path/to/main_heading.tsv, 
                                "dev-cnn-model-2022-v1", 
                                "dev-pointwise-model-2022-v2", 
                                "dev-listwise-model-2022-v2",
                                cnn_batch_size=128,
                                pointwise_batch_size=128,
                                listwise_batch_size=128)

desc_predictions = pipeline.predict(input_data)
```