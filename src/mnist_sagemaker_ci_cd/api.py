"""mnist-sagemaker-ci-cd REST API."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root() -> str:
    """Read root."""
    return "Hello world"
