#!/bin/bash

#SBATCH --job-name=cpu_multiproc_example
#SBATCH -o outputs/cpu_multiproc_example_%j.out
#SBATCH -e outputs/cpu_multiproc_example_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --cpus-per-task=4     # Request 4 CPUs for multiprocessing
#SBATCH --time=00:10:00

#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples/

source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
conda activate env_genoa

python cpu_multiproc_example.py

conda deactivate
