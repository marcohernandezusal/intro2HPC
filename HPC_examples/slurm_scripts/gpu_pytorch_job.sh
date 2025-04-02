#!/bin/bash

#SBATCH --job-name=gpu_pytorch_example
#SBATCH -o outputs/gpu_pytorch_example_%j.out
#SBATCH -e outputs/gpu_pytorch_example_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

#SBATCH -D /path/to/your/project

source /home/your_user/miniforge3/etc/profile.d/conda.shP
conda activate your_gpu_env

python gpu_pytorch_example.py

conda deactivate
