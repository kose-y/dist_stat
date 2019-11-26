import torch
import argparse
import os
from dist_stat.pet_utils import *
from dist_stat.pet_l1_single import PET_L1
dtype = torch.DoubleTensor
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sps

def coo_to_sparsetensor(spm, TType=torch.DoubleTensor):
    typename = torch.typename(TType).split('.')[-1]
    TType_cuda = TType.is_cuda
    densemodule = torch.cuda if TType_cuda else torch
    spmodule = torch.cuda.sparse if TType_cuda else torch.sparse
    TType_sp = getattr(spmodule, typename)
    i = densemodule.LongTensor(np.vstack([spm.row, spm.col]))
    v = TType(spm.data)
    return TType_sp(i, v, spm.shape)

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
        #floatlib.set_ftz()
        #floatlib.set_daz()
    rank = 0



    datafile = np.load("../data/pet_100_120.npz")

    n_x = datafile['n_x']
    n_t = datafile['n_t']
    e = sps.coo_matrix(datafile['e'])
    # e = torch.Tensor(datafile['e']).type(TType)
    e = coo_to_sparsetensor(e, TType)

    TType_name = torch.typename(TType).split('.')[-1]
    TType_sp   = getattr(torch.sparse, TType_name)
    if args.with_gpu:
        TType_sp   = getattr(torch.cuda.sparse, TType_name)
    print(TType_sp)


    D_values = torch.Tensor(datafile['D_values']).type(TType)
    D_indices = torch.Tensor(datafile['D_indices']).type(TType).long()
    D_shape = datafile['D_shape']
    D_size  = torch.Size([int(D_shape[0]), int(D_shape[1])])

    D = TType_sp(D_indices, D_values, D_size)

    counts = torch.Tensor(datafile['counts']).type(TType)

    pet = PET_L1(counts, e, D, sig = 1/3, tau = 1/3, rho=1e-2, TType=TType)
    pet.run(check_obj=True, tol=1e-9)
