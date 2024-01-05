"""This module is used to define the inference functions for the trained model."""
import json
import logging
import os

import torch
from bertopic import BERTopic

JSON_CONTENT_TYPE = "application/json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    """This function is used to load model artifacts from the model_dir directory.

    Args:
        model_dir (str): The directory where model files are stored in S3 as a .tar.gz file.
    """
    logger.info(f"inside model_fn, model_dir= {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device Type: {device}")
    model = BERTopic.load(os.path.join(model_dir, "my_model"))
    return model


def predict_fn(data, model):
    """This function is used to perform inference on a single data point.

    Args:
        data (list): The input data in the format of a list of strings received from the output of the input_fn() function.
        model (object): The model object loaded in memory by model_fn()
    """
    logger.info(f"Got input Data: {data}")
    return model.transform(data)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    """This function is used to deserialize the input data into a Python object like a list or a dictionary.

    Args:
        serialized_input_data (str): The serialized input data in the content_type format (in this case: list of text)
        content_type (str): The content type of the serialized input data.
    """
    logger.info(f"serialized_input_data object: {serialized_input_data}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        logger.info(f"input_data object: {input_data}")
        return input_data
    raise Exception(f"{content_type} not supported by script!")


def output_fn(prediction, content_type):
    """This function is used to serialize the inference output into a JSON.

    Args:
        prediction (list): The output of the predict_fn() function.
        content_type (str): The content type of the serialized input data.
    """
    logger.info(f"prediction object before: {prediction}, type: {type(prediction)}")
    prediction = list(prediction)
    logger.info(f"prediction object after: {prediction}, type: {type(prediction)}")
    prediction[0] = [int(res_class) for res_class in prediction[0]]
    logger.info(f"prediction[0] object after: {prediction[0]}, type: {type(prediction[0])}")
    prediction[1] = [float(res_class) for res_class in prediction[1]]
    logger.info(f"prediction[1] object after: {prediction[0]}, type: {type(prediction[1])}")
    prediction_result = {
        "predictions": prediction[0],
        "scores": prediction[1],
    }
    prediction_result = json.dumps(prediction)
    logger.info(f"prediction_result object: {prediction_result}")
    return prediction_result
