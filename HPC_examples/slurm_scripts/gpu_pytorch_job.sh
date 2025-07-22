#!/bin/bash

#SBATCH --job-name=gpu_pytorch_example
#SBATCH -o outputs/gpu_pytorch_example_%j.out
#SBATCH -e outputs/gpu_pytorch_example_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples/


source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
module unuse `module use`
module use /soft/genoa/EB/modules/all
module load /soft/genoa/EB/modules/all/CUDA/12.2.0

conda activate env_genoa

python gpu_pytorch_example.py

conda deactivate
