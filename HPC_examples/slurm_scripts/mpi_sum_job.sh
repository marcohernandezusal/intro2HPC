#!/bin/bash

#SBATCH --job-name=mpi_sum
#SBATCH --output=outputs/mpi_sum_%j.out
#SBATCH --error=outputs/mpi_sum_%j.err

#SBATCH --partition=short
#SBATCH --ntasks=4                 # 4 MPI processes
#SBATCH --nodes=1                  # 1 node
#SBATCH --time=00:05:00
#SBATCH -D /path/to/your/project

module load mpi                   # Load MPI module (adjust to your cluster)
source /home/your_user/miniforge3/etc/profile.d/conda.sh
conda activate your_mpi_env       # Environment with mpi4py installed

srun python mpi_parallel_sum.py

conda deactivate
