#!/bin/bash

#SBATCH --job-name=cpu_multiproc_example
#SBATCH -o outputs/cpu_multiproc_example_%j.out
#SBATCH -e outputs/cpu_multiproc_example_%j.err

#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --cpus-per-task=4     # Request 4 CPUs for multiprocessing
#SBATCH --time=00:10:00

#SBATCH -D /path/to/your/project

source /home/your_user/miniforge3/etc/profile.d/conda.sh
conda activate your_cpu_env

python cpu_multiproc_example.py

conda deactivate
