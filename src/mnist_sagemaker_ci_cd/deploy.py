"""This module contains the code for deploying the mdoel to Sagemaker following training."""
import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import IdentitySerializer

from mnist_sagemaker_ci_cd.lib.settings import Settings

# Sagemaker Variables
SESSION = sagemaker.Session(boto3.Session(region_name="us-east-1"))

# Github and AWS Settings
SETTINGS = Settings()

# Get the model path from the training job
model_path = Estimator.attach(
    training_job_name=SETTINGS.short_sha, sagemaker_session=SESSION
).model_data

# Define the Model
model = PyTorchModel(
    model_data=model_path,  # type: ignore
    role=SETTINGS.iam_role,
    sagemaker_session=SESSION,
    source_dir="./src/mnist_sagemaker_ci_cd/lib/",
    framework_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    name=SETTINGS.short_sha,
    code_location=SETTINGS.output_s3_uri,
)

# Deploy the Model
endpoint_name = f"{SETTINGS.short_sha}"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    serializer=IdentitySerializer(),
    deserializer=JSONDeserializer(),
    endpoint_logging=True,
)

# Create a CML Runner Comment
message = (
    ":rocket: The model has been deployed to Sagemaker! :rocket:\n\n"
    "## Testing Model\n\n"
    "To test the model, run the following command:\n\n"
    "```python\n"
    "import json\n"
    "import boto3\n"
    "sagemaker_runtime = boto3.client('sagemaker-runtime')\n"
    "with open('../data/4.png', 'rb') as f:\n"
    "    payload = f.read()\n"
    "    try:\n"
    "        response = sagemaker_runtime.invoke_endpoint(\n"
    f"            EndpointName='{SETTINGS.short_sha}',\n"
    "            ContentType='application/octet-stream',\n"
    "            Accept='application/json',\n"
    "            Body=payload\n"
    "        )\n"
    "    except Exception as e:\n"
    "        raise(e)\n"
    "json_body = json.loads(response['Body'].read().decode('utf-8'))\n"
    "json_body['prediction'][0]\n"
    "```\n\n"
    "## Notebooks\n\n"
    "You can also refer to the notebook [here](./notebooks/invoke-endpoint.ipynb) for more details. :smile: \n\n"
    "## Deleting the Endpoint\n\n"
    "To delete the endpoint, run the following command:\n\n"
    "```python\n"
    f"predictor.delete_endpoint(EndpointName='{SETTINGS.short_sha}')\n"
    "```\n\n"
)

message_str = "".join(message)

# Create a CML Runner Comment
with open("details.txt", "w") as f:
    f.write(message_str)
