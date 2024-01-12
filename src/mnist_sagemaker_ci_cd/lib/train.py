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
import json
import logging
import os
import subprocess
import sys

import horovod.torch as hvd
import torch.nn.functional as F
import torch.utils.data.distributed
import wandb
from torch import nn, optim
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def retrieve_data():
    """Retrieve data from Data Version Control.

    This function retrieves the data from DVC and moves it to the input directory.
    It is advisable not to keep this unless you are not using DVC.

    This function moves the contents of the data/ folder using the *.* wildcard to move all subfolders and files.
    """
    # Data Version Control
    commands = [
        "dvc pull",
        "cp -rv data/* /opt/ml/input/data/",
        "rm -rf data",
        "ls -hR /opt/ml/input/data/",
    ]

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Check if the command was executed successfully
        if process.returncode != 0:
            print(f"Error executing '{cmd}': {stderr.decode()}")
        else:
            print(f"Output of '{cmd}': {stdout.decode()}")


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


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data sampler and data loader")
    dataset = datasets.MNIST(
        training_dir,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )
    return train_loader


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data sampler and data loader")
    dataset = datasets.MNIST(
        training_dir,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs
    )
    return test_loader


def train(args):
    """Train the model."""
    logger.debug(f"Number of gpus available - {args.num_gpus}")

    # Horovod: initialize library
    hvd.init()
    torch.manual_seed(args.seed)

    rank = hvd.rank()
    # DVC - Retrieve Data from Remote.
    if rank == 0:
        logger.info("\nRetrieving Data from DVC.\n")
        logger.info(f"\nData Directory: {args.data_dir}\n")
        retrieve_data()

    # Horovod: pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    # Horovod: limit number of CPU threads to be used per worker
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net()

    wandb.watch(model)

    lr_scaler = hvd.size()

    model.cuda()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                wandb.log({"training_loss": loss.item()})
        test(model, test_loader)
    save_model(model, args.model_dir)
    wandb.run.finish()


def _metric_average(val, name):
    """Compute the average over all workers for a metric tracked by horovod."""
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def test(model, test_loader):
    """Validate the model."""
    model.eval()
    test_loss = 0
    test_accuracy = 0
    examples = []
    predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()
            
            # Log example data and predictions
            examples.append(data.cpu().detach().numpy())
            predictions.append(pred.cpu().detach().numpy())

    # Horovod: use test_sampler to determine the number of examples in this worker's partition.
    test_loss /= len(test_loader.sampler)
    test_accuracy /= len(test_loader.sampler)

    # Horovod: average metric values across workers.
    test_loss = _metric_average(test_loss, "avg_loss")
    test_accuracy = _metric_average(test_accuracy, "avg_accuracy")
    
    # Log example data and predictions using wandb
    examples = np.concatenate(examples, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {100 * test_accuracy:.2f}%\n")
    wandb.log({"examples": [wandb.Image(img) for img in examples], "predictions": predictions})

def save_model(model, model_dir):
    """Save the model."""
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container Environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/")
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    logger.info("\nStarting Training.\n")
    wandb.init(
        id=os.environ["WANDB_RUN_ID"],
        project="mnist-sagemaker",
        entity="bhargava-ajay",
        resume="must",
    )
    train(parser.parse_args())
