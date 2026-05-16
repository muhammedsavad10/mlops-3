import sagemaker
from sagemaker.sklearn import SKLearnModel
import time
import boto3


role="arn:aws:iam::477010600979:role/service-role/AmazonSageMaker-ExecutionRole-20260511T140775"

boto_session=boto3.Session(region_name="eu-north-1")
session=sagemaker.Session(boto_session=boto_session)


endpoint_name=f"iris-endpoint-{int(time.time())}"

model=SKLearnModel(
    model_data="s3://my-ml-model-buckets/model.tar.gz",
    role=role,
    entry_point="inference.py",
    sagemaker_session=session,
    framework_version="1.2-1"
)


sm_client=boto3.client("sagemaker",region_name="eu-north-1")

predicted=model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)
