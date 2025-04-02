import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse

def setup_logging(rank):
    logging.basicConfig(
        filename=f"logs/ddp_rank{rank}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

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

def train(rank, world_size, args):
    setup_logging(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=(rank==0), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = SimpleCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    logging.info(f"Rank {rank} training with {args.epochs} epochs.")

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        total_loss = 0
        ddp_model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Rank {rank}, Epoch {epoch}, Loss: {total_loss:.4f}")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size)

if __name__ == "__main__":
    main()
