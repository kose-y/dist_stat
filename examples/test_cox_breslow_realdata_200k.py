import torch
import torch.distributed as dist
import numpy as np
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat.distmat import THDistMat
from dist_stat import distmm

from dist_stat import cox_breslow as cox
import argparse
import os
from pandas_plink import read_plink1_bin
import dask.array as da
num_gpu = torch.cuda.device_count()

from numpy import genfromtxt

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="cox testing")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False, 
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--normalize', dest='normalize', action='store_const', const=True, default=False, 
                        help='normalize the dataset')
    parser.add_argument('--tol', dest='tol', action='store', default=0.0, 
                        help='relative tolerance')
    parser.add_argument('--lambda', dest='lambd', action='store', default=0.0001, 
                        help='penalty parameter')
    parser.add_argument('--iter', dest='iter', action='store', default=10000, 
                        help='max iter')
    parser.add_argument('--nosubnormal', dest='nosubnormal', action='store_const', const=True, default=False, 
                        help='use this flag to avoid subnormal number.')
    parser.add_argument('--step', dest='step', action='store',  default=100, 
                        help='evaluation intervals')
    parser.add_argument('--out-prefix', dest='oprefix', action='store', help="output prefix")

    #parser.add_argument('--rows', dest='rows', action='store', default=10000)
    #parser.add_argument('--cols', dest='cols', action='store', default=10000)
    parser.add_argument('--datnormest', dest='datnormest', action='store', default=None)
    parser.add_argument('--quicknorm', dest='quicknorm', action='store_const', const=True, default=False)
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
    if args.nosubnormal:
        torch.set_flush_denormal(True)
        #floatlib.set_ftz()
        #floatlib.set_daz()
    
    #n = int(args.cols); p = int(args.rows)
    bedfile = "/shared/ukbiobank_filtered/filtered_200k.bed"
    famfile = "/shared/ukbiobank_filtered/filtered_200k.2.fam"

    G = read_plink1_bin(bedfile, fam=famfile, verbose=False)

    n = G.shape[0]
    p_pheno = 11
    p = G.shape[1] + 6

    start_ind = (p // size) * rank
    end_ind   = (p // size) * (rank + 1) 
    pheno = genfromtxt("/shared/ukbiobank_filtered/ukb_short.filtered.200k.tab", skip_header=1)

    if rank != size - 1:
        X_chunk = G[:, start_ind:end_ind].data.compute()
    else:
        X_chunk = da.hstack([G[:,start_ind:].data, da.zeros((n, 6))]).compute()
        X_chunk[:, -11:] = pheno[:, 1:p_pheno + 1]

    from utils import impute_na
    X_chunk = impute_na(X_chunk)

    # normalize
    if args.normalize:
        X_chunk -= X_chunk.mean(0)
        X_chunk /= X_chunk.std(0)

    X_chunk = torch.tensor(X_chunk)
    print(X_chunk.shape)
    X = THDistMat.from_chunks(X_chunk, force_bycol=True)
    
    time = torch.tensor(pheno[:, 12]).view(-1, 1).type(TType)
    delta = torch.tensor(pheno[:, 13]).view(-1, 1).type(TType)
    print(args.lambd) 

    #X = distmat.distgen_normal(p, n, TType=TType, set_from_master=True)
    #torch.manual_seed(seed+100)
    #delta = torch.multinomial(torch.tensor([1., 1.]), n, replacement=True).float().view(-1, 1).type(TType)
    if args.datnormest:
        cox_driver = cox.COX(X, delta, float(args.lambd), nonsnps=11, time=time, TType=TType, sigma=1.0/(2*float(args.datnormest)**2))
    elif args.quicknorm:
        cox_driver = cox.COX(X, delta, float(args.lambd), nonsnps=11, time=time, TType=TType, sigma='quicknorm')
    else:
        cox_driver = cox.COX(X, delta, float(args.lambd), nonsnps=11, time=time, TType=TType, sigma='power')  
    cox_driver.run(int(args.iter), tol=float(args.tol),check_interval=int(args.step), check_obj=True)
    print((cox_driver.beta!=0).type(torch.int64).sum())
    nonzero_ind = (cox_driver.beta != 0).chunk.reshape(-1).cpu().numpy().astype(np.bool)
    print(np.sum(nonzero_ind))
    n_chunk = cox_driver.beta.chunk.shape[0]
    nonzero_idx = np.arange(n_chunk * rank, n_chunk * (rank + 1))[nonzero_ind]
    nonzero_val = cox_driver.beta.chunk.cpu().numpy().reshape(-1)[nonzero_ind]
    print(nonzero_idx, nonzero_val)
    R = np.hstack([nonzero_idx.reshape(-1,1), nonzero_val.reshape(-1, 1)])
    np.savetxt("{}_{}.txt".format(args.oprefix, rank), R, fmt=["%d", "%e"], delimiter="\t")


