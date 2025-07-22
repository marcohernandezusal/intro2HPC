import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import fcntl

import matplotlib
matplotlib.use("Agg")

# --------------------- MODELO ---------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- AUXILIARES -----------------
def setup_logger(path):
    logger = logging.getLogger(f"logger_{path}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def plot_metrics(losses, accuracies, output_dir):
    epochs = range(1, len(losses) + 1)
    plt.figure()
    plt.plot(epochs, losses)
    plt.title("Training Loss")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, accuracies)
    plt.title("Test Accuracy")
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

def write_csv(csv_path, job_id, lr, bs, acc, model_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.writer(f)
        writer.writerow([job_id, lr, bs, acc, model_path])
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ----------------- ENTRENAMIENTO -----------------
def train(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_dir = os.path.join("mnist_outputs", f"job_{args.job_id}_lr{args.lr}_bs{args.batch_size}")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger(os.path.join(output_dir, "train.log"))
        logger.info(f"Rank {rank}/{world_size} | Using device: {device}")
        logger.info(f"Hyperparams: LR={args.lr}, BS={args.batch_size}, Epochs={args.epochs}")
    else:
        logger = None

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("mnist_data", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST("mnist_data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    losses = []
    accuracies = []

    for epoch in range(1, args.epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = ddp_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            acc = evaluate(ddp_model.module, test_loader, device)
            losses.append(total_loss)
            accuracies.append(acc)
            logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    if rank == 0:
        model_path = os.path.join(output_dir, "ddp_model.pth")
        torch.save(ddp_model.module.state_dict(), model_path)
        plot_metrics(losses, accuracies, output_dir)
        write_csv("mnist_outputs/ddp_results.csv", args.job_id, args.lr, args.batch_size, accuracies[-1], model_path)
        logger.info(f"Saved model and metrics to {output_dir}")

    dist.destroy_process_group()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--job_id", type=str, default="0")
    args = parser.parse_args()

    start = datetime.now()
    train(args)
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Finished. Total time: {datetime.now() - start}")
