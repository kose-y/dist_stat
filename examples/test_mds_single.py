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
    parser.add_argument('--tol', dest='tol', action='store', default=0, 
                        help='error tolerance')
    parser.add_argument('--offset', dest='offset', action='store', default=0, 
                        help='gpu id offset')
    parser.add_argument('--datapoints', dest='datapoints', action='store', default=10000)
    parser.add_argument('--origdims', dest='origdims', action='store', default=10000)
    parser.add_argument('--iter', dest='iter', action='store', default=10000)
    parser.add_argument('--targetdims', dest='targetdims', action='store', default=20)
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

    m = torch.DoubleTensor(int(args.datapoints), int(args.origdims)).uniform_().type(TType)
    
    mds_driver = mds.MDS(m, int(args.targetdims), TType=TType)
    mds_driver.run(int(args.iter), tol=float(args.tol),check_interval=100, check_obj=True)

