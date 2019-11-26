import torch
import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat import distmm
from dist_stat import nmf
import argparse
import os
if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    num_gpu=4
else:
    num_gpu=8


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="nmf testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_true', 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_true',
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--nosubnormal', dest='nosubnormal', action='store_true',
                        help='use this flag to avoid subnormal number.')
    parser.add_argument('--tol', dest='tol', action='store', default=0, 
                        help='error tolerance')
    parser.add_argument('--rows', dest='m', action='store', default=10000,
                        help='number of rows')
    parser.add_argument('--cols', dest='n', action='store', default=10000,
                        help='number of cols')
    parser.add_argument('--r', dest='r', action='store', default=20,
                        help='internal dim')
    parser.add_argument('--iter', dest='iter', action='store', default=10000,
                        help='max iter')
    args = parser.parse_args()
    if args.with_gpu:
        divisor = size//num_gpu
        if divisor==0:
            torch.cuda.set_device(rank)
        else:
            torch.cuda.set_device(rank//divisor)
        if args.double:
            TType=torch.cuda.DoubleTensor
        else:
            TType=torch.cuda.FloatTensor
    else:
        if args.double:
            TType=torch.DoubleTensor
        else:
            TType=torch.FloatTensor
    if args.nosubnormal:
        torch.set_flush_denormal(True)
        #floatlib.set_ftz()
        #floatlib.set_daz()

    torch.manual_seed(95376+rank)

    m = distmat.distgen_uniform(int(args.m), int(args.n), TType=TType, set_from_master=False)
    nmf_driver = nmf.NMF(m, int(args.r), TType, init_from_master=False)
    nmf_driver.run(int(args.iter), tol=float(args.tol), check_interval=100, check_obj=False)
