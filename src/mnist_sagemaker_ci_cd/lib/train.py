"""This module contains the training logic for BERTtopic."""
# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import ast
import logging
import os
import subprocess
import sys

import torch
from bertopic import BERTopic

JSON_CONTENT_TYPE = "application/json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def retrieve_data():
    """Retrieve data from Data Version Control.

    This function retrieves the data from DVC and moves it to the input directory.
    It is advisable not to keep this unless you are not using DVC.

    This function moves the contents of the data/ folder using the *.* wildcard to move all subfolders and files.
    """
    # Data Version Control
    commands = ["dvc pull", "mv data/*.* /opt/ml/input/data/"]

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Check if the command was executed successfully
        if process.returncode != 0:
            print(f"Error executing '{cmd}': {stderr.decode()}")
        else:
            print(f"Output of '{cmd}': {stdout.decode()}")


def _train(args, data_dir="/opt/ml/input/data"):
    """Train the model based on the training data."""
    logger.debug("BERTtopic training starting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Type: {device}")

    model = BERTopic(language=args.language)

    print(f"BERTtopic Model loaded for language {args.language}")

    print("Loading Training data")
    print(f"data_dir: {data_dir}")
    with open(data_dir + "/training_data.txt") as file:
        docs = [line.rstrip() for line in file]
    print(f"Training data loaded. Number of documents: {len(docs)}")
    print("Started Training")
    topics, probs = model.fit_transform(docs)
    print("Finished Training")
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    print("Saving the model.")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    path = os.path.join(model_dir, "my_model")
    model.save(
        path, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # custom parameter for BERTopic
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help='main language for the input documents. If you want a multilingual model that supports 50+ languages, select "multilingual".',
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework. When running locally, you can set these environment variables.

    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    retrieve_data()
    _train(parser.parse_args(), data_dir="/opt/ml/input/data")

    sys.exit()
