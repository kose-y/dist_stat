import torch
from dist_stat import mds_single as mds
import argparse
import os



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="nmf testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False, 
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--nosubnormal', dest='nosubnormal', action='store_const', const=True, default=False, 
                        help='use this flag to avoid subnormal number.')
    args = parser.parse_args()
    if args.with_gpu:
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
    rank = 0
    torch.manual_seed(95376+rank)

    m = torch.DoubleTensor(10000, 1000).uniform_().type(TType)
    
    mds_driver = mds.MDS(m, 100, TType=TType)
    mds_driver.run(100000, tol=0,check_interval=1, check_obj=True)

