"""This module contains the settings for the SageMaker Script."""
import os

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """A class that represents the settings for the SageMaker Script.

    Attributes:
                github_ref_name (str): The name of the GitHub reference (branch or tag) being used. Defaults to "development-training".
                github_sha (str): The SHA of the GitHub commit being used. Defaults to "1930a1baa305519c96d186903644216fda399ecc".
                github_repository (str): The URL of the GitHub repository. Defaults to "https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd.git".
                github_actor (str): The username of the GitHub actor (user or bot) triggering the pipeline. Defaults to "ajay-bhargava".
                github_pat (str): The GitHub Personal Access Token (PAT) used for authentication. Defaults to a sample token.
                estimator_dataset_s3_uri (str): The S3 URI for the estimator dataset.
                output_s3_uri (str): The S3 URI for the output of the pipeline.

    Methods:
    check_s3_uri(cls, v): A validator method that checks if the provided URI is a valid S3 URI.

    Config:
    env_file (str): The name of the environment file to load configuration from.
    """

    github_ref_name: str = os.environ.get("GITHUB_REF_NAME", "dvc-training-branch")
    github_sha: str = os.environ.get("GITHUB_SHA", "e4ff64acf6d3fa466d8304df8bdf17409a0292bb")
    github_repository: str = os.environ.get(
        "GITHUB_REPOSITORY", "https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd.git"
    )
    github_actor: str = os.environ.get("GITHUB_ACTOR", "ajay-bhargava")
    github_pat: str = os.environ.get(
        "GITHUB_PAT",
        "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",
    )
    estimator_dataset_s3_uri: str = os.environ.get(
        "S3_BASE_PATH", "s3://with-context-sagemaker/datasets/bert-topic/data/"
    )
    output_s3_uri: str = f"s3://with-context-sagemaker/fits/bert-topic/{github_ref_name}/"

    @validator("estimator_dataset_s3_uri", "output_s3_uri")
    @classmethod
    def check_s3_uri(cls, v):
        """A validator method that checks if the provided URI is a valid S3 URI."""
        if "s3://" not in v:
            raise ValueError("The URI must be a valid S3 URI")
        return v
