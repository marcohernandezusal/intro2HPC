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
    logging.info("PyTorch job started.")
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of CUDA devices available: {num_gpus}")

        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            logging.info(f"GPU {i}: {device_name}")

        device = torch.device("cuda:0")
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logging.warning("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

    # Dummy workload to simulate tensor operations
    logging.info("Running dummy tensor operation...")
    x = torch.rand((5000, 5000), device=device)
    y = torch.mm(x, x)
    result = y.sum().item()

    logging.info(f"Computation complete. Result: {result:.4f}")
    logging.info("PyTorch job finished.")

if __name__ == "__main__":
    main()
