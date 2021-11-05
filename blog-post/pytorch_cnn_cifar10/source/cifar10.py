import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from tqdm import tqdm

from sagemaker_tune import num_gpu
from sagemaker_tune.report import Reporter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class Net(nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super(Net, self).__init__()
        assert 0 <= dropout_rate <= 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    for inputs, labels in tqdm(data_loader):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            prediction = model(inputs)
            prediction = prediction.max(1)[1]
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
    n_valid = len(data_loader.sampler)
    percentage_correct = 100.0 * correct / n_valid
    return percentage_correct / 100


def _train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    n_train = len(trainset) * 8 // 10
    n_val = len(trainset) - n_train
    train_split, val_split = torch.utils.data.random_split(trainset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(
        train_split, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_split, batch_size=128, shuffle=True, num_workers=args.workers
    )

    logger.info(f"length training/validation splits: {len(train_split)}/{len(val_split)}")
    logger.info("Model loaded")
    model = Net(dropout_rate=args.dropout_rate)

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    reporter = Reporter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch, i + 1, running_loss / 2000))
                running_loss = 0.0
        val_acc = compute_accuracy(model=model, data_loader=val_loader, device=device)
        reporter(epoch=epoch, val_acc=val_acc)
    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        metavar="DR",
        help="dropout rate (default: 0.0)",
    )

    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )

    parser.add_argument("--model-dir", type=str, default=os.environ.get('SM_MODEL_DIR', "./"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get('SM_CHANNEL_TRAINING', "./data/"),
        help="the folder containing cifar-10-batches-py/",
    )
    parser.add_argument("--num-gpus", type=int, default=num_gpu())

    args, _ = parser.parse_known_args()

    _train(args)
