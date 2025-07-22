#!/bin/bash

#SBATCH --job-name=mnist_ddp
#SBATCH --output=outputs/mnist_ddp_%j.out
#SBATCH --error=outputs/mnist_ddp_%j.err

#SBATCH --partition=genoa
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH -D /mnt/lustre_fs/_HPC/SCRATCH/usal_bisite_1/usal_bisite_1_2/Marco/SCAYLE_examples

# Activar entorno y módulos
source /home/usal_bisite_1/COMUNES/miniforge3/etc/profile.d/conda.sh
module unuse `module use`
module use /soft/genoa/EB/modules/all
module load /soft/genoa/EB/modules/all/CUDA/12.2.0
conda activate env_genoa

# Variables para debug y compatibilidad
export NCCL_IB_DISABLE=1                    # Desactiva Infiniband si no disponible
export NCCL_DEBUG=INFO                      # Más logs de NCCL
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Más logs de torch.distributed


# Ejecutar entrenamiento
torchrun --nproc_per_node=4 train_ddp.py --lr 0.001 --batch_size 64 --epochs 5 --job_id test01

conda deactivate
