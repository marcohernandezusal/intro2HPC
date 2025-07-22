#!/bin/bash

#############################################
#           Slurm Job Script Guide          #
#############################################

# ---- Job Identification ----

#SBATCH --job-name=my_job_name         # Name of the job (shows in queue)
#SBATCH --output=outputs/%x_%j.out     # Standard output (%x = job name, %j = job ID)
#SBATCH --error=outputs/%x_%j.err      # Standard error log

# ---- Partition & QoS ----

#SBATCH --partition=genoa              # Partition (queue) to run the job on
#SBATCH --qos=normal                   # Quality of Service (e.g. normal, high, debug)

# ---- Resource Requests ----

#SBATCH --nodes=1                      # Number of nodes (entire machines)
#SBATCH --ntasks=1                     # Total number of tasks (MPI processes)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task (OpenMP or multiprocessing)
#SBATCH --gres=gpu:2                   # Generic resources (e.g., 2 GPUs)
#SBATCH --mem=16G                      # Memory per node (or per task if --mem-per-cpu used)

# ---- Time Management ----

#SBATCH --time=01:00:00                # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL                # Email on: BEGIN, END, FAIL, etc.
#SBATCH --mail-user=your@email.com     # Your email address for notifications

# ---- Job Arrays ----

#SBATCH --array=0-4                    # Job array of 5 jobs (IDs 0 to 4)
# Use $SLURM_ARRAY_TASK_ID in your script to access current job's ID

# ---- Output Directory ----

#SBATCH -D /home/user/project          # Working directory for job

# ---- Logging & Debugging ----

#SBATCH --open-mode=append             # Append to output files (instead of overwrite)
#SBATCH --requeue                     # Allow job to be requeued after node failure
#SBATCH --signal=B:USR1@60             # Send signal before timeout (for cleanup/checkpointing)

#############################################
#         Your Job Starts Here              #
#############################################

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Running in: $(pwd)"

# Example conda environment activation
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate my_env

# Example command (could be python, mpirun, torchrun, etc.)
python my_script.py --config config_${SLURM_ARRAY_TASK_ID}.yaml

# Deactivate env when done
conda deactivate
