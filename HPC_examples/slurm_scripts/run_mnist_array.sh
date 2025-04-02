#!/bin/bash

#SBATCH --job-name=mnist_hyperparam
#SBATCH --output=outputs/mnist_%A_%a.out
#SBATCH --error=outputs/mnist_%A_%a.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --time=00:20:00
#SBATCH --array=0-5

#SBATCH -D /path/to/your/project

source /home/your_user/miniforge3/etc/profile.d/conda.sh
conda activate your_gpu_env

# Define hyperparameters per task ID
lrs=(0.01 0.001 0.0001 0.005 0.0005 0.00005)
batches=(32 64 128 64 128 256)

lr=${lrs[$SLURM_ARRAY_TASK_ID]}
batch_size=${batches[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with lr=$lr, batch_size=$batch_size"
python train_mnist.py --lr $lr --batch_size $batch_size --epochs 5

conda deactivate
