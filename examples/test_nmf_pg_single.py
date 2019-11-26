import torch
from dist_stat import nmf_pg_single as nmf
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
    parser.add_argument('--rows', dest='m', action='store', default=10000,
                        help='number of rows')
    parser.add_argument('--cols', dest='n', action='store', default=10000,
                        help='number of cols')
    parser.add_argument('--r', dest='r', action='store', default=20,
                        help='internal dim')
    #parser.add_argument('--eps', dest='eps', action='store', default=1e-6,
    #                    help='ridge penalty')
    parser.add_argument('--iter', dest='iter', action='store', default=10000,
                        help='max iter')
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
        #floatlib.set_ftz()
        #floatlib.set_daz()
    rank = 0
    torch.manual_seed(95376+rank)

    m = TType(int(args.m), int(args.n)).uniform_()
    
    nmf_driver = nmf.NMF(m, int(args.r), TType)
    nmf_driver.run(int(args.iter), tol=float(args.tol),check_interval=100, check_obj=True)
