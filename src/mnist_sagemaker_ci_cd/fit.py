"""Module description goes here."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

from mnist_sagemaker_ci_cd.lib.settings import Settings

settings = Settings()

# AWS Variables
SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))
IAM_ROLE = "arn:aws:iam::220582896887:role/programmatic-aws-sagemaker-role-access"
ACCOUNT_ID = SESSION.boto_session.client("sts").get_caller_identity()["Account"]
TRAINING_INSTANCE = "ml.g4dn.xlarge"

# Hyperparameters
hyperparameters: dict[str, str] = {
    "language": "english",
}

# Define Estimator
estimator = Estimator(
    image_uri="220582896887.dkr.ecr.us-east-1.amazonaws.com/mlops-sagemaker:latest",
    role=IAM_ROLE,
    instance_count=1,
    entry_point="src/mnist_sagemaker_ci_cd/lib/train.py",
    instance_type=TRAINING_INSTANCE,
    hyperparameters=hyperparameters,  # type: ignore
    base_job_name=settings.output_s3_uri,
    output_path=settings.output_s3_uri,
    code_location=settings.output_s3_uri,
    sagemaker_session=SESSION,
    source_dir="./",
    dependencies=["src/mnist_sagemaker_ci_cd/deps/train/requirements.txt"],
)

estimator.fit(job_name=f"{settings.github_sha[:7]}")
