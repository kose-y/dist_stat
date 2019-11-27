import torch
import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat import distmm
import argparse
import os
from dist_stat.euclidean_distance import euclidean_distance_DistMat, euclidean_distance_tensor
num_gpu = torch.cuda.device_count()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="nmf testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False, 
                        help='use this flag for double precision. otherwise single precision is used.')
    args = parser.parse_args()
    if args.with_gpu:
        torch.cuda.set_device(rank % num_gpu)
        if args.double:
            TType=torch.cuda.DoubleTensor
        else:
            TType=torch.cuda.FloatTensor
    else:
        if args.double:
            TType=torch.DoubleTensor
        else:
            TType=torch.FloatTensor

    data = distmat.distgen_normal(10000, 1000, TType=TType)
    r = euclidean_distance_DistMat(data)

    print(r.chunk)

    if rank==0:
        print(euclidean_distance_tensor(data.chunk, data.chunk))
