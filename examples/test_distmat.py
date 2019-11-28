import os
import torch
import torch.distributed as dist
import platform
import argparse

dist.init_process_group('mpi')

from dist_stat import distmat
from dist_stat.distmm import *
rank = dist.get_rank()
size = dist.get_world_size()
num_gpu = torch.cuda.device_count()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Examples of distributed matrix operations")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False,
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False,
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--offset', dest='offset', action='store', default=0,
                        help='gpu id offset')
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
    distmat_1 = distmat.distgen_normal(12, 12, TType=TType)
    print ("r ", dist.get_rank(), distmat_1.chunk)

    if rank==0:
        p = 12; q = 12
        data = TType(p,q).normal_()
        print(data)
    else:
        data = None

    rslt = distmat.dist_data(data, src=0, TType=TType)

    print(rank, rslt.chunk)

    r1 = rslt.diag()

    print(r1.chunk)

    r2 = rslt.diag(distribute=False)
    print(r2)

    rr = rslt + rslt

    print(rank, rslt.chunk, rr.chunk)

    dist.barrier()

    rslt += rslt
    print("rslt += rslt", rank, rslt.chunk)

    print((1 + rslt).chunk)
    print((rslt + torch.arange(12, out=TType(1,12))).chunk)
    # print((torch.arange(12, out=TType(1,12)) + rslt).chunk) # doesn't work

    if rank==0:
        col0 = data[:, 0].view(-1, 1)
    else:
        col0 = None
    if rank==0:
        print(col0)
    col0_dist = distmat.dist_data(col0, src=0, TType=TType)
    print("adding by 1st col of data", rank, (rslt + col0_dist).chunk)
    rslt.fill_diag_(0)
    print("rslt, diagonal set to zero", rank, rslt.chunk)


