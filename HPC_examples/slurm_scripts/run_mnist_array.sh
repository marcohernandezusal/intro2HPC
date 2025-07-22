#!/bin/bash

#SBATCH --job-name=mnist_hyperparam
#SBATCH --output=outputs/mnist_%A_%a.out
#SBATCH --error=outputs/mnist_%A_%a.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --time=00:20:00

#SBATCH --array=0-17

#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples/

# Load environment
source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
module unuse `module use`
module use /soft/genoa/EB/modules/all
module load /soft/genoa/EB/modules/all/CUDA/12.2.0
conda activate env_genoa

# Define hyperparameter ranges
lrs=(0.01 0.001 0.0001 0.005 0.0005 0.00005)
batches=(32 64 128)

# Compute total combinations
total_lrs=${#lrs[@]}
total_batches=${#batches[@]}
total_combinations=$((total_lrs * total_batches))

# Validate array bounds
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range."
    exit 1
fi

# Map index to hyperparams
lr_index=$((SLURM_ARRAY_TASK_ID / total_batches))
batch_index=$((SLURM_ARRAY_TASK_ID % total_batches))
lr=${lrs[$lr_index]}
batch_size=${batches[$batch_index]}

# Create human-readable job identifier
job_id="lr${lr}_bs${batch_size}"

echo "Running job $SLURM_ARRAY_TASK_ID with lr=$lr, batch_size=$batch_size"
echo "Job ID: $job_id"

python train_mnist.py --lr $lr --batch_size $batch_size --epochs 5 --job_id $SLURM_ARRAY_TASK_ID

conda deactivate
