# MTIX Descriptor Prediction Pipeline

## Installation

1. Create a Python 3.10 virtual environment. 
2. Install the package:

```
pip install .
```
3. Upload the CNN, Pointwise, and Listwise model files to an AWS S3 bucket.
4. Deploy the SageMaker endpoints using the script provided in the git repository (./scripts/create_sagemaker_endpoints.py). Deployment settings (e.g. S3 bucket name) can be modified at the top of the script. The script prints endpoint names for the deployed models.

## Test
Automated unit and integration tests can be run using pytest. To run the integration test you will need to update the SageMaker endpoint names in the integration test code. The integration test checks predictions for 40k citations, and it may therefore take a long time to run.
```
pytest -m unit
pytest -m integration
```

## Usage

The pipeline is constructed with 5 input parameters:

1. Path to the Descriptor name lookup file. This file maps internal Descriptor ids to Descriptor names.
2. Path to Desciptor unique identifier file. This file maps internal Descriptor ids to NLM DUIs.
3. Sagemaker endpoint name for CNN model.
4. Sagemaker endpoint name for Pointwise model.
5. Sagemaker endpoint name for Listwise model.

```
from mtix_descriptor_prediction_pipeline import create_descriptor_prediction_pipeline

pipeline = create_descriptor_prediction_pipeline("path/to/main_heading_names.tsv", 
                                                 "path/to/main_heading.tsv, 
                                                 "tensorflow-inference-2022-04-01-22-15-17-484", 
                                                 "huggingface-pytorch-inference-2022-04-01-22-18-14-890", 
                                                 "huggingface-pytorch-inference-2022-04-01-22-21-50-717")

desc_predictions = pipeline.predict(input_data)
```
