"""Module description goes here."""
import boto3
import sagemaker
import wandb
from sagemaker.estimator import Estimator

from mnist_sagemaker_ci_cd.lib.settings import Settings

# Github and AWS Settings
settings = Settings()

# Sagemaker Variables
SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))
TRAINING_INSTANCE = "ml.g4dn.xlarge"

# W&B Variables
id = wandb.util.generate_id()  # type: ignore
wandb.init(id=id, project=settings.github_repo_name, entity="bhargava-ajay", resume="must")

# Hyperparameters
hyperparameters = {
    "epochs": 10,
    "backend": "gloo",
}

# Environment Variables
environment = {
    "WANDB_RUN_ID": id,
    "WANDB_PROJECT": settings.github_repo_name,
    "WANDB_RUN_GROUP": settings.github_ref_name,
}

# Define Estimator
estimator = Estimator(
    image_uri=settings.ecr_repo_name,
    role=settings.iam_role,
    entry_point="src/mnist_sagemaker_ci_cd/lib/train.py",
    instance_type=TRAINING_INSTANCE,
    instance_count=2,
    environment=environment,
    hyperparameters=hyperparameters,  # type: ignore
    base_job_name=settings.output_s3_uri,
    output_path=settings.output_s3_uri,
    code_location=settings.output_s3_uri,
    sagemaker_session=SESSION,
    source_dir="./",
    dependencies=["src/mnist_sagemaker_ci_cd/deps/fit/requirements.txt"],
)

estimator.fit(job_name=f"{settings.github_sha[:7]}", wait=False, logs="Training")
wandb.run.finish(quiet=True)  # type: ignore
