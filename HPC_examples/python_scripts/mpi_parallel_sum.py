from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Root process creates a large array
if rank == 0:
    data = np.arange(1, 100001)  # Sum of 1 to 100,000
    print(f"[Rank {rank}] Total elements: {len(data)}")
else:
    data = None

# Step 2: Scatter the data evenly
chunk_size = 100000 // size
local_data = np.zeros(chunk_size, dtype='i')
comm.Scatter([data, MPI.INT], [local_data, MPI.INT], root=0)

# Step 3: Each process computes the local sum
local_sum = np.sum(local_data)
print(f"[Rank {rank}] Local sum: {local_sum}")

# Step 4: Reduce (gather and sum) all local sums at root
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Step 5: Root prints the final result
if rank == 0:
    print(f"[Rank {rank}] Total sum = {total_sum}")
