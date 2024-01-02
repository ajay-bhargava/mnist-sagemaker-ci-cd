"""Module description goes here."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

from mnist_sagemaker_ci_cd.lib.settings import Settings

session = sagemaker.Session(boto3.Session(region_name="us-east-1"))
settings = Settings()

# AWS Variables
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
ACCOUNT_ID = session.boto_session.client("sts").get_caller_identity()["Account"]
TRAINING_INSTANCE = "ml.g4dn.xlarge"

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
    base_job_name=settings.output_s3_uri,
    output_path=settings.output_s3_uri,
    code_location=settings.output_s3_uri,
    sagemaker_session=session,
    source_dir="src/mnist_sagemaker_ci_cd/",
    dependencies=["src/mnist_sagemaker_ci_cd/deps/train/requirements.txt"],
    git_config={
        "repo": settings.github_repository,
        "branch": settings.github_ref_name,
        "username": settings.github_actor,
        "token": settings.github_pat,
    },
)

estimator.fit(settings.estimator_dataset_s3_uri, job_name=f"{settings.github_sha[:7]}")
