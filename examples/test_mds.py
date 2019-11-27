import torch
import argparse
import os

import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat import mds

if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    num_gpu=4
else:
    num_gpu=8

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="nmf testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False, 
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--nosubnormal', dest='nosubnormal', action='store_const', const=True, default=False, 
                        help='use this flag to avoid subnormal number.')
    parser.add_argument('--tol', dest='tol', action='store', default=0, 
                        help='error tolerance')
    parser.add_argument('--offset', dest='offset', action='store', default=0, 
                        help='gpu id offset')
    parser.add_argument('--datapoints', dest='datapoints', action='store', default=10000)
    parser.add_argument('--origdims', dest='origdims', action='store', default=10000)
    parser.add_argument('--iter', dest='iter', action='store', default=10000)
    parser.add_argument('--targetdims', dest='targetdims', action='store', default=20)
    parser.add_argument('--set_from_master', dest='set_from_master', action='store_true', 
                        help='samples are generated from the CPU of root: for obtaining identical dataset for different settings.')
    args = parser.parse_args()
    if args.with_gpu:
        divisor = size//num_gpu
        if divisor==0:
            torch.cuda.set_device(rank+int(args.offset))
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

    m = distmat.distgen_normal(int(args.datapoints), int(args.origdims), set_from_master=args.set_from_master)
    
    mds_driver = mds.MDS(m, int(args.targetdims), TType=TType, init_from_master=args.set_from_master)
    mds_driver.run(int(args.iter), tol=float(args.tol),check_interval=100, check_obj=True)

