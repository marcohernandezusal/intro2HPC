#!/bin/bash
#SBATCH --job-name=summarize_mnist
#SBATCH --output=outputs/summarize.out
#SBATCH --error=outputs/summarize.err
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --time=00:01:00
#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples/

# Load environment
source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
conda activate env_genoa

# Run summary script
python find_best_mnist_model.py

conda deactivate