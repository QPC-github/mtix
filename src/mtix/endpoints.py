import boto3
import math
import os.path
from sagemaker.exceptions import ObjectNotExistedError
import time
import uuid


class HuggingFaceEndpointHelper:

    def process_request(self, request):
        inputs = request["inputs"]
        parameters = request["parameters"]
        return inputs, parameters

    def construct_batch_data(self, batch_inputs, parameters):
        batch_data = { "inputs": batch_inputs, "parameters": parameters}
        return batch_data

    def process_response(self, response):
        return response

    def construct_output(self, result_list):
        return result_list


class TensorflowEndpointHelper:

    def process_request(self, request):
        inputs = request["instances"]
        parameters = None
        return inputs, parameters

    def construct_batch_data(self, batch_inputs, _):
        batch_data = {"instances": batch_inputs}
        return batch_data

    def process_response(self, response):
        result = response["predictions"]
        return result

    def construct_output(self, result_list):
        output = { "predictions": result_list}
        return output


class RealTimeEndpoint:

    def __init__(self, endpoint_helper, sagemaker_rt_endpoint, batch_size):
        self.helper = endpoint_helper
        self.sagemaker_rt_endpoint = sagemaker_rt_endpoint
        self.batch_size = batch_size
    
    def predict(self, request):
        inputs, parameters = self.helper.process_request(request)

        input_count = len(inputs)
        num_batches = int(math.ceil(input_count/self.batch_size))

        result_list = []
        for idx in range(num_batches):
            batch_start = idx * self.batch_size
            batch_end = (idx + 1) * self.batch_size
            batch_inputs = inputs[batch_start:batch_end]
            batch_data = self.helper.construct_batch_data(batch_inputs, parameters)
            response = self.sagemaker_rt_endpoint.predict(batch_data)
            result = self.helper.process_response(response)
            result_list.extend(result)
        
        predictions = self.helper.construct_output(result_list)
        return predictions


class AsyncEndpoint:

    def __init__(self, endpoint_helper, sagemaker_async_endpoint, endpoint_name, bucket_name, temp_dir, batch_size, wait_time):
        self.helper = endpoint_helper
        self.sagemaker_async_endpoint = sagemaker_async_endpoint
        self.endpoint_name = endpoint_name
        self.bucket_name = bucket_name
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.s3 = boto3.client("s3")

    def predict(self, request):
        inputs, parameters = self.helper.process_request(request)

        input_count = len(inputs)
        num_batches = int(math.ceil(input_count/self.batch_size))

        response_list = []
        input_key_lookup = {}
        output_key_lookup = {}
        for idx in range(num_batches):
            batch_start = idx * self.batch_size
            batch_end = (idx + 1) * self.batch_size
            batch_inputs = inputs[batch_start:batch_end]
            batch_data = self.helper.construct_batch_data(batch_inputs, parameters)
            batch_uuid = str(uuid.uuid4())
            batch_input_key = os.path.join(self.temp_dir, self.endpoint_name, "inputs", batch_uuid)
            input_key_lookup[idx] = batch_input_key
            batch_input_path = os.path.join(f"s3://{self.bucket_name}", batch_input_key)
            batch_response = self.sagemaker_async_endpoint.predict_async(data=batch_data, input_path=batch_input_path)
            response_list.append(batch_response)
            batch_output_file = os.path.basename(batch_response.output_path)
            output_key = os.path.join(self.temp_dir, self.endpoint_name, "outputs", batch_output_file)
            output_key_lookup[idx] = output_key

        result_lookup = {}
        while len(result_lookup) < num_batches:
            time.sleep(self.wait_time)
            for idx in range(num_batches):
                if idx not in result_lookup:
                    try:
                        batch_response = response_list[idx]
                        batch_result = batch_response.get_result()
                        result_lookup[idx] = batch_result
                        self.try_delete(input_key_lookup[idx])
                        self.try_delete(output_key_lookup[idx])
                    except ObjectNotExistedError:
                        continue
  
        result_list = []
        for idx in range(num_batches):
            result = result_lookup[idx]
            result = self.helper.process_response(result)
            result_list.extend(result)

        predictions = self.helper.construct_output(result_list)

        return predictions

    def try_delete(self, key):
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
        except:
            pass
        

class HuggingFaceRealTimeEndpoint(RealTimeEndpoint):
    def __init__(self, sagemaker_rt_hf_endpoint, batch_size=128):
        super().__init__(HuggingFaceEndpointHelper(), sagemaker_rt_hf_endpoint, batch_size)


class TensorflowRealTimeEndpoint(RealTimeEndpoint):
    def __init__(self, sagemaker_rt_tf_endpoint, batch_size=128):
        super().__init__(TensorflowEndpointHelper(), sagemaker_rt_tf_endpoint, batch_size)


class HuggingFaceAsyncEndpoint(AsyncEndpoint):
    def __init__(self, sagemaker_async_hf_endpoint, endpoint_name, bucket_name, temp_dir="async_inference", batch_size=128, wait_time=0.01):
        super().__init__(HuggingFaceEndpointHelper(), sagemaker_async_hf_endpoint, endpoint_name, bucket_name, temp_dir, batch_size, wait_time)


class TensorflowAsyncEndpoint(AsyncEndpoint):
    def __init__(self, sagemaker_async_tf_endpoint, endpoint_name, bucket_name, temp_dir="async_inference", batch_size=128, wait_time=0.01):
        super().__init__(TensorflowEndpointHelper(), sagemaker_async_tf_endpoint, endpoint_name, bucket_name, temp_dir, batch_size, wait_time)