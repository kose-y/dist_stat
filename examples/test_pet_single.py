import torch
import argparse
import os
from dist_stat.pet_utils import *
from dist_stat.pet_single import PET
dtype = torch.DoubleTensor
import numpy as np



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
    e = torch.Tensor(datafile['e']).type(TType)

    TType_name = torch.typename(TType).split('.')[-1]
    TType_sp   = getattr(torch.sparse, TType_name)
    if args.with_gpu:
        TType_sp   = getattr(torch.cuda.sparse, TType_name)
    print(TType_sp)

    G_values = torch.Tensor(datafile['G_values']).type(TType)
    G_indices = torch.Tensor(datafile['G_indices']).type(TType).long()
    G_shape = datafile['G_shape']
    G_size  = torch.Size([int(G_shape[0]), int(G_shape[1])])

    G = TType_sp(G_indices, G_values, G_size)
    print(G)

    D_values = torch.Tensor(datafile['D_values']).type(TType)
    D_indices = torch.Tensor(datafile['D_indices']).type(TType).long()
    D_shape = datafile['D_shape']
    D_size  = torch.Size([int(D_shape[0]), int(D_shape[1])])

    D = TType_sp(D_indices, D_values, D_size)

    counts = torch.Tensor(datafile['counts']).type(TType)

    pet = PET(counts, e, G, D, mu=1e-6, TType=TType)
    pet.run(check_obj=True, tol=1e-9)
