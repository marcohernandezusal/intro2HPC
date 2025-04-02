import torch
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename="logs/gpu_pytorch_example.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("PyTorch GPU job started.")
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Exiting.")
        return
    
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of CUDA devices available: {num_gpus}")

    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        logging.info(f"GPU {i}: {device_name}")

    device = torch.device("cuda:0")

    logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    # Dummy workload to simulate GPU usage
    logging.info("Running dummy tensor operation on GPU...")
    x = torch.rand((5000, 5000), device=device)
    y = torch.mm(x, x)
    result = y.sum().item()

    logging.info(f"Computation complete. Result: {result:.4f}")
    logging.info("PyTorch GPU job finished.")

if __name__ == "__main__":
    main()
