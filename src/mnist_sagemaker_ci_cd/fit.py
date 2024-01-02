"""Module description goes here."""
import os

import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto3.Session(region_name="us-east-1"))

# AWS Variables
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
ACCOUNT_ID = session.boto_session.client("sts").get_caller_identity()["Account"]
TRAINING_INSTANCE = "ml.g4dn.xlarge"

# Github Actions Variables
GITHUB_REF_NAME = os.environ.get("GITHUB_REF_NAME", "development-training")
GITHUB_SHA = os.environ.get("GITHUB_SHA", "1927b1baa305519c96d186903644216fda399ecc")
GITHUB_REPOSITORY = os.environ.get(
    "GITHUB_REPOSITORY", "https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd.git"
)
GITHUB_ACTOR = os.environ.get("GITHUB_ACTOR", "ajay-bhargava")
GITHUB_PAT = os.environ.get(
    "GITHUB_PAT",
    "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",
)

# Dataset Variables (Change this to DVC Repro later)
ESTIMATOR_DATASET_S3_URI = "s3://with-context-sagemaker/datasets/bert-topic/"
OUTPUT_S3_URI = f"s3://with-context-sagemaker/fits/bert-topic/{GITHUB_REF_NAME}/"

# Hyperparameters
hyperparameters: dict[str, str] = {
    "language": "english",
}

# Define Estimator
estimator = Estimator(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310",
    role=IAM_ROLE,
    instance_count=1,
    entry_point="lib/train.py",
    instance_type=TRAINING_INSTANCE,
    hyperparameters=hyperparameters,  # type: ignore
    base_job_name=OUTPUT_S3_URI,
    output_path=OUTPUT_S3_URI,
    code_location=OUTPUT_S3_URI,
    sagemaker_session=session,
    source_dir="src/mnist_sagemaker_ci_cd/",
    dependencies=["fit-requirements.txt"],
    git_config={
        "repo": GITHUB_REPOSITORY,
        "branch": GITHUB_REF_NAME,
        "username": GITHUB_ACTOR,
        "token": GITHUB_PAT,
    },
)

estimator.fit(ESTIMATOR_DATASET_S3_URI, job_name=f"{GITHUB_SHA[:7]}")
