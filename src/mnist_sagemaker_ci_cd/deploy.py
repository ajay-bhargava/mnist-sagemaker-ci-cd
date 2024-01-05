"""This module contains the code for deploying the mdoel to Sagemaker following training."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.model import Model

from mnist_sagemaker_ci_cd.lib.settings import Settings

SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
SETTINGS = Settings()

model_path = Estimator.attach(
    training_job_name=SETTINGS.github_sha[:7], sagemaker_session=SESSION
).model_data

model = Model(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker",
    model_data=model_path,
    role=IAM_ROLE,
    sagemaker_session=SESSION,
    source_dir="./",
    entry_point="src/mnist_sagemaker_ci_cd/lib/inference.py",
    dependencies=["src/mnist_sagemaker_ci_cd/deps/fit/requirements.txt"],
    name=SETTINGS.github_sha[:7],
)


endpoint_name = "1940a1c-predictor"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
)

print(f"\nDeployed model to endpoint: {endpoint_name}")
