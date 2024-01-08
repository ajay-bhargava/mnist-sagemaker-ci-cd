"""REST API."""
try:
    import unzip_requirements
except ImportError:
    pass
import io
from ast import literal_eval
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile
from mangum import Mangum
from PIL import Image
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

ENDPOINT = "8qgf6d3"

sample_number = """[
 [
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 84 , 185, 159, 151, 60 , 36 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52 , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 67 , 114, 72 , 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 17 , 66 , 14 , 67 , 67 , 67 , 59 , 21 , 236, 254, 106, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 83 , 253, 209, 18 , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 22 , 233, 255, 83 , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 129, 254, 238, 44 , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 59 , 249, 254, 62 , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 133, 254, 187, 5  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 9  , 205, 248, 58 , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 126, 254, 182, 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 75 , 251, 240, 57 , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 19 , 221, 254, 166, 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 3  , 203, 254, 219, 35 , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 38 , 254, 254, 77 , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 31 , 224, 254, 115, 1  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 133, 254, 254, 52 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 61 , 242, 254, 254, 52 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 121, 254, 254, 219, 40 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 121, 254, 207, 18 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0, 0, 0, 0, 0, 0]
 ],
]"""


def run_inference(file: Any):
    """Run an inference by converting an image to numpy array, then resize it to (1,1,28,28) then send to Sagemaker Endpoint."""
    image = Image.open(io.BytesIO(file))
    new_image = image.resize((28, 28))
    grayscale_image = new_image.convert("L")
    image = np.array(grayscale_image, dtype=np.float32)
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        serializer=NumpySerializer(),
        deserializer=JSONDeserializer(),
    )
    return predictor.predict(image)


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
    return "Hello world"


@app.get("/demo")
def predict() -> dict:
    """Predict."""
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        serializer=NumpySerializer(),
        deserializer=JSONDeserializer(),
    )
    example = literal_eval(sample_number)
    image = np.array([example], dtype=np.float32)
    return predictor.predict(image)


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
