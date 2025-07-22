#!/bin/bash

#SBATCH --job-name=cpu_logging_example
#SBATCH -o outputs/cpu_logging_example_%j.out
#SBATCH -e outputs/cpu_logging_example_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1                  # 1 task
#SBATCH --cpus-per-task=2     # Use 2 CPUs
#SBATCH --time=00:10:00       # 10 minutes

# Set working directory
#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples/

# Load any necessary modules or activate conda
source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
conda activate env_genoa

# Run the Python script
python cpu_logging_example.py

conda deactivate
