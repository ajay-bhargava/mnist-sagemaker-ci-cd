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
wandb.init(id=id, project=settings.github_repo_name, entity="bhargava-ajay")
wandb_run_url = wandb.run.get_url()  # type: ignore

# Hyperparameters
hyperparameters = {
    "epochs": 10,
    "backend": "gloo",
}

# Environment Variables
environment = {
    "WANDB_RUN_ID": id,
    "WANDB_API_KEY": settings.wandb_api_key,
    "WANDB_PROJECT": settings.github_repo_name,
    "WANDB_RUN_GROUP": settings.github_sha[:7],
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

# Create a CML Runner Comment
message = (
    f":crystal_ball: Hi! Sagemaker training launch detected. :rocket: \n\n",
    f"You can view the details of this run in the table by clicking on the link below.\n\n",
    f"| Item | Value |\n",
    f"| --- | --- |\n",
    f"| Job Name | {settings.github_sha[:7]} |\n",
    f"| Training Instance | {TRAINING_INSTANCE} |\n",
    f"| W&B :sparkles: Job URL | [Here]({wandb_run_url}) |\n",
    f"| S3 Artifacts | [Here]({settings.output_s3_uri}) |\n",
    f"| Training Logs | [Here]({settings.output_s3_uri}) |\n"
    f"\n\n"
)

# Write the comment to the PR
with open("details.txt", "w") as f:
    f.write(message)