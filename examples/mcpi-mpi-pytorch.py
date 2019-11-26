import torch.distributed as dist
import torch
dist.init_process_group('mpi') # initialize MPI

rank = dist.get_rank()         # device id
size = dist.get_world_size()   # total number of devices

# select device
device = 'cuda:{}'.format(rank) # or simply 'cpu' for CPU computing
if device.startswith('cuda'): torch.cuda.set_device(rank)

def mc_pi(n):
    # this code is executed on each device.
    x = torch.rand((n), dtype=torch.float64, device=device)
    y = torch.rand((n), dtype=torch.float64, device=device)
    # compute local estimate of pi
    r = torch.mean((x**2 + y**2 < 1).to(dtype=torch.float64))*4
    dist.all_reduce(r) # sum of 'r's in each device is stored in 'r'
    return r / size

if __name__ == '__main__':
    n = 10000
    r = mc_pi(n)
    if rank == 0:
        print(r.item())
