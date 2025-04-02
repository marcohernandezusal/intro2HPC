import torch
import logging
import subprocess
import os
from datetime import datetime

# Setup logging
log_filename = "logs/gpu_debug.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return e.stdout.decode(), e.stderr.decode()

def log_environment():
    logging.info("========== ENVIRONMENT ==========")
    logging.info(f"User: {os.getenv('USER')}")
    logging.info(f"PATH: {os.getenv('PATH')}")
    logging.info(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
    logging.info(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
    logging.info(f"Python Executable: {os.sys.executable}")
    logging.info(f"Datetime: {datetime.now()}")

def log_torch_info():
    logging.info("========== TORCH INFO ==========")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available in torch: {torch.cuda.is_available()}")
    logging.info(f"Compiled with CUDA: {torch.version.cuda}")
    logging.info(f"CUDNN version: {torch.backends.cudnn.version()}")

def log_nvidia_smi():
    logging.info("========== NVIDIA-SMI ==========")
    out, err = run_cmd("nvidia-smi")
    logging.info(f"nvidia-smi stdout:\n{out}")
    if err:
        logging.error(f"nvidia-smi stderr:\n{err}")

def log_nvcc_version():
    logging.info("========== NVCC VERSION ==========")
    out, err = run_cmd("nvcc --version")
    logging.info(f"nvcc stdout:\n{out}")
    if err:
        logging.error(f"nvcc stderr:\n{err}")

def check_cuda_devices():
    logging.info("========== CUDA DEVICES ==========")
    count = torch.cuda.device_count()
    logging.info(f"Number of CUDA devices: {count}")
    if count == 0:
        logging.warning("No CUDA devices found by PyTorch.")
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            logging.info(f"GPU {i}: {name} - Compute Capability: {cap}")
        except Exception as e:
            logging.error(f"Error accessing GPU {i}: {e}")

def dummy_gpu_workload():
    logging.info("========== DUMMY GPU WORKLOAD ==========")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, skipping workload.")
        return
    try:
        device = torch.device("cuda:0")
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        x = torch.rand((2048, 2048), device=device)
        y = torch.mm(x, x)
        result = y.sum().item()
        logging.info(f"Dummy computation successful. Result: {result:.4f}")
    except Exception as e:
        logging.error(f"Error during dummy GPU workload: {e}")

def main():
    logging.info("========== DEBUG START ==========")
    log_environment()
    log_torch_info()
    log_nvidia_smi()
    log_nvcc_version()
    check_cuda_devices()
    dummy_gpu_workload()
    logging.info("========== DEBUG END ==========")

if __name__ == "__main__":
    main()
