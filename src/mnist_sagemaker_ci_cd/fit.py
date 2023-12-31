"""Module description goes here."""
import os

import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.LocalSession()

# AWS Variables
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
ACCOUNT_ID = session.boto_session.client("sts").get_caller_identity()["Account"]
TRAINING_INSTANCE = "local"

# Github Actions Variables
GITHUB_REF_NAME = os.environ.get("GITHUB_REF_NAME", "development-training")
GITHUB_SHA = os.environ.get("GITHUB_SHA", "db056f755716e8a38704402b1bed37a0c0136c48")
GITHUB_REPOSITORY = os.environ.get(
    "GITHUB_REPOSITORY", "https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd.git"
)
GITHUB_ACTOR = os.environ.get("GITHUB_ACTOR", "ajay-bhargava")
OUTPUT_S3_URI = f"s3://with-context-sagemaker/fits/bert-topic/{GITHUB_REF_NAME}/{GITHUB_SHA}"
GITHUB_PAT = os.environ.get(
    "GITHUB_PAT",
    "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",
)

# Dataset Variables (Change this to DVC Repro later)
ESTIMATOR_DATASET_S3_URI = "s3://with-context-sagemaker/datasets/bert-topic/"

# Hyperparameters
hyperparameters: dict[str, str] = {
    "language": "english",
}

# Define Estimator
estimator = Estimator(
    image_uri=f"{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/sagemaker-ecr-bert-topic-example:latest",  # 220582896887.dkr.ecr.us-east-1.amazonaws.com/sagemaker-ecr-bert-topic-example:latest
    role=IAM_ROLE,
    instance_count=1,
    instance_type=TRAINING_INSTANCE,
    hyperparameters=hyperparameters,  # type: ignore
    output_path=OUTPUT_S3_URI,
    entry_point="src/mnist_sagemaker_ci_cd/fit.py",  # "src/mnist_sagemaker_ci_cd/train.py
    sagemaker_session=session,
    git_config={
        "repo": GITHUB_REPOSITORY,
        "branch": GITHUB_REF_NAME,
        "username": GITHUB_ACTOR,
        "token": GITHUB_PAT,
    },
)

estimator.fit(ESTIMATOR_DATASET_S3_URI)
