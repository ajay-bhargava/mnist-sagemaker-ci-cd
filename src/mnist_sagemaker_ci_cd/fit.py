"""Module description goes here."""
import os

import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto3.Session(region_name="us-east-1"))

# AWS Variables
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
ACCOUNT_ID = session.boto_session.client("sts").get_caller_identity()["Account"]
TRAINING_INSTANCE = "ml.m5.large"

# Github Actions Variables
GITHUB_REF_NAME = os.environ.get("GITHUB_REF_NAME", "/repository/heads/main")
GITHUB_SHA = os.environ.get("GITHUB_SHA", "db056f755716e8a38704402b1bed37a0c0136c48")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "ajay-bhargava/mnist-sagemaker-ci-cd")
GITHUB_ACTOR = os.environ.get("GITHUB_ACTOR", "ajay-bhargava")
OUTPUT_S3_URI = (
    f"s3://with-context-sagemaker-examples/fits/bert-topic/{GITHUB_REF_NAME}/{GITHUB_SHA}"
)
GITHUB_PAT = os.environ.get(
    "GITHUB_PAT",
    "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",
)

# Dataset Variables (Change this to DVC Repro later)
ESTIMATOR_DATASET_S3_URI = "s3://with-context-sagemaker-examples/datasets/bert-topic/"

# Hyperparameters
hyperparameters: dict[str, str] = {
    "language": "english",
}

# Define Estimator
estimator = Estimator(
    image_uri=f"{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/with-context-sagemaker-examples:latest",
    role=IAM_ROLE,
    instance_count=1,
    instance_type=TRAINING_INSTANCE,
    hyperparameters=hyperparameters,  # type: ignore
    output_path=OUTPUT_S3_URI,
    sagemaker_session=session,
    git_config={
        "repo": GITHUB_REPOSITORY,
        "branch": GITHUB_REF_NAME,
        "username": GITHUB_ACTOR,
        "token": GITHUB_PAT,
    },
)
