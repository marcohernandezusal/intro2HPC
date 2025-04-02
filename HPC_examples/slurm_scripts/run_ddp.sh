#!/bin/bash

#SBATCH --job-name=mnist_ddp
#SBATCH --output=outputs/mnist_ddp_%j.out
#SBATCH --error=outputs/mnist_ddp_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --gres=gpu:4           # use all 4 GPUs on one node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

#SBATCH -D /path/to/your/project

source /home/your_user/miniforge3/etc/profile.d/conda.sh
conda activate your_gpu_env

# torchrun is preferred over launch since PyTorch 1.9+
torchrun --nproc_per_node=4 train_ddp.py --epochs 5 --lr 0.001 --batch_size 64

conda deactivate
