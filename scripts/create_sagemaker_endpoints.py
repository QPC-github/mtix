from sagemaker.huggingface import HuggingFaceModel
from sagemaker.tensorflow import TensorFlowModel


INSTANCE_TYPE = "ml.m5.xlarge"
S3_BUCKET = "s3://raear-hf-sagemaker-inference"
CNN_MODEL_FILENAME = "cnn_model_2022_v1.tar.gz"
POINTWISE_MODEL_FILENAME = "pointwise_model_2022_v1.tar.gz"
LISTWISE_MODEL_FILENAME = "listwise_model_2022_v1.tar.gz"


def create_cnn_model_endpoint():
    model = TensorFlowModel(
        model_data=f"{S3_BUCKET}/{CNN_MODEL_FILENAME}",
        role="SageMakerExecute",
        framework_version="2.8.0",
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE
    )
    print("CNN model endpoint name:")
    print(predictor.endpoint_name)


def create_pointwise_model_endpoint():
    huggingface_model = HuggingFaceModel(
        model_data=f"{S3_BUCKET}/{POINTWISE_MODEL_FILENAME}",
        role="SageMakerExecute",
        transformers_version="4.12.3",
        pytorch_version="1.9.1",
        py_version="py38",
        env={ "HF_TASK": "text-classification" }
    )
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE
    )
    print("Pointwise model endpoint name:")
    print(predictor.endpoint_name)


def create_listwise_model_endpoint():
    huggingface_model = HuggingFaceModel(
        model_data=f"{S3_BUCKET}/{LISTWISE_MODEL_FILENAME}",
        role="SageMakerExecute",
        transformers_version="4.12.3",
        pytorch_version="1.9.1",
        py_version="py38",
        env={ "HF_TASK": "token-classification" }
    )
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE
    )
    print("Listwise model endpoint name:")
    print(predictor.endpoint_name)


def main():
    create_cnn_model_endpoint()
    create_pointwise_model_endpoint()
    create_listwise_model_endpoint()


if __name__ == "__main__":
    main()