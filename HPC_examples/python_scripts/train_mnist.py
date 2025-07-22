# train_mnist_offline.py

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv
import fcntl

import matplotlib
matplotlib.use("Agg")


def write_results_to_csv(csv_path, job_id, lr, batch_size, accuracy, model_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        # Lock file to avoid race condition
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        writer = csv.writer(csvfile)
        writer.writerow([job_id, lr, batch_size, accuracy, model_path])
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)


# Simple CNN model
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

def plot_loss_accuracy(train_losses, test_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

def plot_sample_predictions(model, test_loader, device, output_dir):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"))
    plt.close()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def setup_logger(log_file: str):
    """Creates a job-specific logger."""
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-imported
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Optional: also log to stdout
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Job-specific output directory
    output_dir = os.path.join("mnist_outputs", f"job_{args.job_id}_lr{args.lr}_bs{args.batch_size}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(output_dir, "train.log")
    logger = setup_logger(log_file)

    logger.info(f"Training started on {device}")
    logger.info(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, job_id={args.job_id}")

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    data_root = "mnist_data"
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"MNIST data not found in '{data_root}'.")

    train_set = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_accuracies = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save model and plots
    model_path = os.path.join(output_dir, "simple_cnn_mnist.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    plot_loss_accuracy(train_losses, test_accuracies, output_dir)
    plot_sample_predictions(model, test_loader, device, output_dir)
    logger.info("Plots saved.")
    logger.info("Training completed.")

    results_csv = "mnist_outputs/grid_results.csv"
    write_results_to_csv(
        csv_path=results_csv,
        job_id=args.job_id,
        lr=args.lr,
        batch_size=args.batch_size,
        accuracy=test_accuracies[-1],
        model_path=model_path
    )
    logger.info(f"Final accuracy ({test_accuracies[-1]:.4f}) logged to {results_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--job_id", type=str, default="0")
    args = parser.parse_args()

    start = datetime.now()
    train(args)
    logging.info(f"Total time: {datetime.now() - start}")
