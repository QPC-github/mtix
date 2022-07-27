# MTIX

## Installation

1. Create a Python 3.9 virtual environment. 
2. Install the package:

```
pip install .
```
3. Upload the CNN, Pointwise, Listwise, and Subheading model files to an AWS S3 bucket.
4. Deploy the SageMaker endpoints using the script provided in the git repository (./scripts/create_sagemaker_endpoints.py). Deployment settings (e.g. S3 bucket name) can be modified at the top of the script.
5. Create the following S3 directories for async prediction temporary files:<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/subheading_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/subheading_endpoint/outputs/<br>

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
3. Path to subheading name lookup file. This file maps NLM QUIs to subheading names.
4. Sagemaker endpoint name for CNN model.
5. Sagemaker endpoint name for Pointwise model.
6. Sagemaker endpoint name for Listwise model.
7. Sagemaker endpoint name for subheading model.
8. The s3 bucket name (for async prediction temporary data).
9. The s3 prefix (for async prediction temporary data).
10. The cnn model batch size.
11. The pointwise model batch size.
12. The listwise model batch size.
13. The subheading model batch size.

Example usage for async endpoints:

```
from mtix import create_async_pipeline

pipeline = create_async_pipeline("path/to/main_heading_names.tsv", 
                                "path/to/main_heading.tsv, 
                                "path/to/subheading_names.tsv,
                                "raear-cnn-endpoint-2022-v1-async", 
                                "raear-pointwise-endpoint-2022-v2-async", 
                                "raear-listwise-endpoint-2022-v2-async",
                                "raear-all-subheading-cnn-endpoint-2022-v1-async",
                                "ncbi-aws-pmdm-ingest",
                                "async_inference",
                                cnn_batch_size=128,
                                pointwise_batch_size=128,
                                listwise_batch_size=128,
                                subheading_batch_size=128)

predictions = pipeline.predict(input_data)
```

Example usage for real-time endpoints:


```
from mtix import create_real_time_pipeline

pipeline = create_real_time_pipeline("path/to/main_heading_names.tsv", 
                                    "path/to/main_heading.tsv, 
                                    "path/to/subheading_names.tsv,
                                    "raear-cnn-endpoint-2022-v1", 
                                    "raear-pointwise-endpoint-2022-v2", 
                                    "raear-listwise-endpoint-2022-v2",
                                    "raear-all-subheading-cnn-endpoint-2022-v1",
                                    cnn_batch_size=128,
                                    pointwise_batch_size=128,
                                    listwise_batch_size=128,
                                    subheading_batch_size=128)

predictions = pipeline.predict(input_data)
```