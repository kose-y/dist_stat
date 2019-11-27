import torch
import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat

A = distmat.distgen_uniform(4, 4)
B = distmat.distgen_uniform(4, 2, TType=torch.DoubleTensor)
AB = distmat.mm(A, B)
if rank == 0:
    print("AB = ")
print(rank, AB.chunk)
C = (1 + AB).log()
if rank == 0:
    print("log(1 + AB) = ")
print(rank, C.chunk)

