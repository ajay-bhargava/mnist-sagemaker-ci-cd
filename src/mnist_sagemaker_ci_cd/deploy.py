"""This module contains the code for deploying the mdoel to Sagemaker following training."""
import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import IdentitySerializer

from mnist_sagemaker_ci_cd.lib.settings import Settings

SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
SETTINGS = Settings()

model_path = Estimator.attach(
    training_job_name=SETTINGS.github_sha[:7], sagemaker_session=SESSION
).model_data

model = PyTorchModel(
    model_data=model_path,  # type: ignore
    role=IAM_ROLE,
    sagemaker_session=SESSION,
    source_dir="./src/mnist_sagemaker_ci_cd/lib/",
    framework_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    name=SETTINGS.github_sha[:7],
    code_location=SETTINGS.output_s3_uri,
)


endpoint_name = f"{SETTINGS.github_sha[:7]}"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    serializer=IdentitySerializer(),
    deserializer=JSONDeserializer(),
    endpoint_logging=True,
)

print(f"\nDeployed model to endpoint: {endpoint_name}")
