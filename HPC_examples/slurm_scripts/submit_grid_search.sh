#!/bin/bash

#!/bin/bash
#SBATCH --job-name=submit_mnist_jobs
#SBATCH --output=outputs/submit_jobs_%j.out
#SBATCH --error=outputs/submit_jobs_%j.err
#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH -D /mnt/lustre_fs/_HPC/HOME/usal_bisite_1/usal_bisite_1_2/SCAYLE/HPC_examples/

# Load conda and any necessary modules (in case sbatch environment needs it)
source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh

# Submit the training job array and capture its job ID
train_jobid=$(sbatch --parsable run_mnist_array.sh)
echo "Submitted training array job with ID: $train_jobid"

# Submit the summarization job with a dependency on the training array
summary_jobid=$(sbatch --dependency=afterok:$train_jobid summarize_best_model.sh)
echo "Submitted summary job with ID: $summary_jobid (depends on training array completion)"