"""REST API."""
try:
    import unzip_requirements
except ImportError:
    pass
from typing import Any

from fastapi import FastAPI, File, UploadFile
from mangum import Mangum
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

ENDPOINT = "8qgf6d3"


def run_inference(file: Any):
    """Run an inference by passing it to the Sagemaker Endpoint."""
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        serializer=IdentitySerializer(),
        deserializer=JSONDeserializer(),
    )
    return predictor.predict(file)


app = FastAPI(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)


@app.get("/")
def read_root() -> str:
    """Read root."""
    return "You have reached the MNIST Sagemaker Endpoint. Please make a POST request to /predict with a file."


@app.post("/predict")
async def predict_file(file: UploadFile = File(...)) -> dict:
    """Predict file."""
    contents = await file.read()
    inference = run_inference(contents)
    prediction = inference["prediction"][0]
    return {
        "filename": file.filename,
        "prediction": prediction,
    }


handler = Mangum(app)
