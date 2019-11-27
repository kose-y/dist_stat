import torch
from dist_stat import cox_single as cox
import argparse
import os

#import pyximport
#pyximport.install()
#import floatlib


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="nmf testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False, 
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--nosubnormal', dest='nosubnormal', action='store_const', const=True, default=False, 
                        help='use this flag to avoid subnormal number.')
    parser.add_argument('--rows', dest='rows', action='store', default=10000)
    parser.add_argument('--cols', dest='cols', action='store', default=10000)
    parser.add_argument('--iter', dest='iter', action='store', default=10000)
    parser.add_argument('--lambda', dest='lmbda', action='store', default=0.0001)
    parser.add_argument('--tol', dest='tol', action='store', default=0.0,
                        help='relative tolerance')
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
    seed = 95376
    torch.manual_seed(seed+rank)
    n = int(args.cols); p = int(args.rows)
    X = torch.randn((p, n))
    torch.manual_seed(seed+100)
    delta = torch.multinomial(torch.tensor([1., 1.]), n, replacement=True).float().view(-1, 1).type(TType)

    cox_driver = cox.COX(X.t(), delta, float(args.lmbda), TType=TType)  
    cox_driver.run(int(args.iter), tol=float(args.tol),check_interval=100, check_obj=True)
