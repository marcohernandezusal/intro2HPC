# **intro2HPC**: 🧠 HPC, SLURM, and Python Parallel Programming Guide

This guide introduces essential concepts and hands-on examples for **High Performance Computing (HPC)** with **SLURM**, **Python multiprocessing**, **PyTorch**, and **MPI/DDP**.

---

## 📦 Table of Contents

1. [Introduction to SLURM](#introduction-to-slurm)
2. [Common SLURM Commands](#common-slurm-commands)
3. [Writing SLURM Job Scripts](#writing-slurm-job-scripts)
4. [Python Multiprocessing on CPU](#python-multiprocessing-on-cpu)
5. [Running GPU Jobs with PyTorch](#running-gpu-jobs-with-pytorch)
6. [Hyperparameter Sweeps with SLURM Arrays](#hyperparameter-sweeps-with-slurm-arrays)
7. [Distributed Training with PyTorch DDP](#distributed-training-with-pytorch-ddp)
8. [Multiprocessing vs MPI](#multiprocessing-vs-mpi)
9. [Resources and References](#resources-and-references)

---

## 🖥️ Introduction to SLURM

**SLURM** is a workload manager for clusters. It handles:
- Job queuing
- Resource allocation (CPUs, GPUs, memory)
- Scheduling and job arrays

> Jobs are submitted via batch scripts or commands.

---

## 🔧 Common SLURM Commands

| Command | Purpose |
|--------|---------|
| `sbatch job.sh` | Submit a job script |
| `squeue` | Show job queue |
| `squeue -u $USER` | Show your jobs |
| `scancel <jobid>` | Cancel a job |
| `sinfo` | Show available partitions/nodes |
| `srun` | Run a command interactively |
| `sacct` | View past job stats |
| `scontrol show job <id>` | Inspect job details |

---

## 📝 Writing SLURM Job Scripts

### Example structure:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -D /home/user/project

source /path/to/conda.sh
conda activate my_env
python my_script.py
conda deactivate
```

---

## 🧠 Python Multiprocessing on CPU

### `cpu_multiproc_example.py`
- Uses `multiprocessing.Pool` to run tasks on 4 CPU cores.
- Demonstrates logging, CPU core detection, and load simulation.
- Run with this job script:

```bash
#SBATCH --cpus-per-task=4
python cpu_multiproc_example.py
```

---

## ⚙️ Running GPU Jobs with PyTorch

### `gpu_pytorch_example.py`
- Logs available GPUs
- Runs matrix multiplication on GPU

### SLURM script:
```bash
#SBATCH --gres=gpu:1
python gpu_pytorch_example.py
```

---

## 🧪 Hyperparameter Sweeps with SLURM Arrays

### `run_mnist_array.sh` + `train_mnist.py`
- Runs the same training with different learning rates and batch sizes.
- Uses SLURM arrays:
```bash
#SBATCH --array=0-5
```

#### Python receives params like:
```bash
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
python train_mnist.py --lr $lr --batch_size $batch_size
```

---

## 🔄 Distributed Training with PyTorch DDP

### `train_ddp.py`
- Implements multi-GPU training using `torch.distributed` and `DDP`
- Each GPU runs a separate process using `mp.spawn()`

### SLURM script `run_ddp.sh`
```bash
#SBATCH --gres=gpu:4
torchrun --nproc_per_node=4 train_ddp.py --epochs 5
```

---

## 🔬 Multiprocessing vs MPI

| Feature | Multiprocessing | MPI |
|--------|------------------|-----|
| Memory Model | Shared | Distributed |
| Typical Use | One node | Multi-node |
| Library | Python stdlib | `mpi4py`, `OpenMPI` |
| Comm. Method | Queues, shared mem | Explicit send/receive |
| Scalability | Limited | Highly scalable |

### When to Use:
- Use **multiprocessing** for single-node parallelism.
- Use **MPI/DDP** for multi-node or large-scale GPU training.

---

## 📚 Resources and References

- [SLURM official documentation](https://slurm.schedmd.com/documentation.html)
- [PyTorch DDP Tutorial](https://pytorch.org/docs/stable/notes/ddp.html)
- [mpi4py documentation](https://mpi4py.readthedocs.io/)
- Your cluster's user guide or documentation

---

> 🧪 *Tip for learners:* Start by mastering CPU multiprocessing, then move to GPU training, and finally dive into distributed training and job arrays.
