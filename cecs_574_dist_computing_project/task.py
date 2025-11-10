"""cecs-574-dist-computing-project: Model + Data + Train/Eval Functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


# ---------------------------------------------------------------------------
# 📦 Model (ResNet18 adapted for CIFAR-10)
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """ResNet18 modified for CIFAR-10 input resolutions."""

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None, num_classes=10)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 🎚 Data Loading (FederatedDataset)
# ---------------------------------------------------------------------------

fds = None

TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def apply_transforms(batch):
    batch["img"] = [TRANSFORM(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Load partition CIFAR10 data with speed-optimized settings."""
    global fds

    # Validate partition_id is within valid range
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} is out of range [0, {num_partitions-1}]"
        )

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition = partition.train_test_split(test_size=0.2, seed=42)
    partition = partition.with_transform(apply_transforms)

    # Balance between data usage and memory (0.5 = 50% of data)
    subset_ratio = 0.3
    train_size = int(len(partition["train"]) * subset_ratio)
    test_size = int(len(partition["test"]) * subset_ratio)
    train_subset = partition["train"].select(range(train_size))
    test_subset = partition["test"].select(range(test_size))

    # Reduce num_workers to save memory (0 = main process only)
    trainloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_subset, batch_size=batch_size, num_workers=0)

    return trainloader, testloader


# ---------------------------------------------------------------------------
# 🏋️ Training + Evaluation
# ---------------------------------------------------------------------------

def train(model, trainloader, epochs, lr, device):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    # Use Adam with slightly lower learning rate for better convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    running_loss = 0.0
    num_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def test(model, testloader, device):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)

            loss = criterion(outputs, labels).item()
            total_loss += loss

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy
