"""This module contains the code for deploying the mdoel to Sagemaker following training."""
import boto3
import numpy as np
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import NumpySerializer

from mnist_sagemaker_ci_cd.lib.settings import Settings

SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
SETTINGS = Settings()

model_path = Estimator.attach(
    training_job_name=SETTINGS.github_sha[:7], sagemaker_session=SESSION
).model_data

model = PyTorchModel(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker",
    model_data=model_path,
    role=IAM_ROLE,
    sagemaker_session=SESSION,
    source_dir="./",
    entry_point="src/mnist_sagemaker_ci_cd/lib/inference.py",
    name=SETTINGS.github_sha[:7],
    code_location=SETTINGS.output_s3_uri,
)


endpoint_name = f"{SETTINGS.github_sha[:7]}"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    # This is required and must match the input type of the inference.py script
    serializer=NumpySerializer(dtype=np.float32),
    deserializer=JSONDeserializer(),
    # End of the deliberate requirement section
    endpoint_logging=True,
)

print(f"\nDeployed model to endpoint: {endpoint_name}")
