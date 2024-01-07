import io
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sagemaker_inference import encoder
from torch import nn


class Net(nn.Module):
    """Neural network model for image classification."""

    def __init__(self):
        """Initialize the neural network model."""
        super(Net, self).__init__()  # noqa: UP008
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        logging.info("Model loaded.")
    return model.to(device)


def input_fn(request_body, request_content_type):
    """An input_fn that loads a numpy array from the request body.

    Args:
        request_body (io.BytesIO): Input data. This is data that has already been serialized by sagemaker.serializer.NumpySerializer()
        request_content_type (str): Request content type.

    """
    logging.info("Received content.")
    try:
        io_bytes = io.BytesIO(request_body)
        array = np.load(io_bytes)
        array = array.reshape(1, 1, 28, 28)
    except Exception as e:
        logging.info(f"Error: {e}")
    logging.info(f"Input: {array.shape}")
    return torch.from_numpy(array).float()


def predict_fn(input_data, model, context):
    """Preprocess input data and make predictions.

    Args:
        input_data (torch.Tensor): Input data.
        model (torch.nn.Module): PyTorch model.
        context (sagemaker_inference.model_server.context.ModelServerContext): Model server context.

    Returns:
        torch.Tensor: Predictions.
    """
    device = torch.device(
        "cuda:" + str(context.system_properties.get("gpu_id"))
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()
    with torch.no_grad():
        logging.info("Making predictions on input.")
        return model(input_data.to(device)).argmax(1)


def output_fn(prediction, content_type, context):
    """An output_fn that dumps the prediction to a JSON format.

    Args:
        prediction (torch.Tensor): Predictions.
        content_type (str): Response content type (this must be specified as a sagemaker.deserializers.JSONDeserializer() if the output is a dictionary.
        context (sagemaker_inference.model_server.context.ModelServerContext): Model server context.
    """
    output = {"prediction": prediction.detach().cpu().numpy().tolist()}
    logging.info(f"Output: {output}")
    try:
        return encoder.encode(output, content_type)
    except Exception as e:
        logging.info(f"Error: {e}")
