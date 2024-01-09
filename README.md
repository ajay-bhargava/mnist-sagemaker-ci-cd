[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd)

# MNIST Serverless Sagemaker Predictor

The intent of this repository is to define CI/CD workflows for Sagemaker to train a model then deploy it using FASTAPI on App Runner. The further intents of this repository is to demonstrate the combined functionality and joining of the `FastAPI` library, together with `Jupyter`, `DVC`, `Serverless`, and `Sagemaker` to create a CI/CD code based infrstructure method for ML. Github Actions orchestrates the CI/CD process.

## CI / CD

_Sagemaker Training_: To train the model, simply issue a PR. This will trigger the CI/CD process to run the training job on Sagemaker. The model will be saved to S3 as per the Github SHA of the PR. Then, the model report will come back to you in the form of a PR comment. The model report will contain the model accuracy, and the model confusion matrix.

_Sagemaker Lambda Endpoint_: Upon merge of the PR, the model will be deployed to a Lambda endpoint. The Lambda endpoint will be deployed using the Serverless framework. You can then run a query of the endpoint to get a prediction.

## Development

<details open>
<summary>Developing | Environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).
2. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/ajay-bhargava/mnist-sagemaker-ci-cd) to clone this repository in a container volume and create a Dev Container with VS Code.
</details>

<details>
<summary>Developing | Poetry Notes</summary>

- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `poetry add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `poetry.lock`. Add `--group test` or `--group dev` to install a CI or development dependency, respectively.
- Run `poetry update` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`.
- To run the FastAPI server for the Serverless Deployment locally, run `poe api --dev`. To deploy this same endpoint to AWS Lambda, run `sls deploy` from the `/src/serverless/` directory.

</details>
