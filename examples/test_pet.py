import torch
import argparse
import os
from dist_stat.pet_utils import *
from dist_stat.pet import PET
import numpy as np
from scipy import sparse


import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat.distmat import THDistMat
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
    parser.add_argument('--mu', dest='mu', action='store', default=0, 
                        help='penalty parameter')
    parser.add_argument('--offset', dest='offset', action='store', default=0, 
                        help='gpu id offset')
    parser.add_argument('--data', dest='data', action='store', default='../data/pet_100_180.npz',
                        help='data file (.npz)')
    parser.add_argument('--iter', dest='iter', action='store', default=1000, 
                        help='max iter')
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

    rank = dist.get_rank()
    size = dist.get_world_size()



    datafile = np.load(args.data)


    n_x = datafile['n_x']
    n_t = datafile['n_t']
    e = torch.Tensor(datafile['e']).type(TType)

    p = e.shape[1]
    d = e.shape[0]

    p_chunk_size = p//size

    e_chunk = e[:, (rank*p_chunk_size):((rank+1)*p_chunk_size)]
    e_dist = THDistMat.from_chunks(e_chunk, force_bycol=True)
    print(e_dist.byrow)
    print(e_dist.shape)

    TType_name = torch.typename(TType).split('.')[-1]
    TType_sp   = getattr(torch.sparse, TType_name)
    if args.with_gpu:
        TType_sp   = getattr(torch.cuda.sparse, TType_name)
    print(TType_sp)


    G_coo = sparse.coo_matrix((datafile['G_values'], 
                                (datafile['G_indices'][0,:], datafile['G_indices'][1,:])), 
                                shape=datafile['G_shape'])
    G_csr = G_coo.tocsr()
    G_csr_chunk = G_csr[(rank*p_chunk_size):((rank+1)*p_chunk_size), :]
    G_coo_chunk = G_csr_chunk.tocoo()
    G_values = TType(G_coo_chunk.data).type(TType)
    G_rows   = torch.LongTensor(G_coo_chunk.row)
    G_cols   = torch.LongTensor(G_coo_chunk.col)
    if G_values.is_cuda:
        G_rows = G_rows.cuda()
        G_cols = G_cols.cuda()
    G_indices = torch.stack([G_rows, G_cols], dim=1).t()
    G_shape = G_coo_chunk.shape
    G_size  = torch.Size([int(G_shape[0]), int(G_shape[1])])
    G_chunk = TType_sp(G_indices, G_values, G_size)
    G_dist  = THDistMat.from_chunks(G_chunk)



    D_coo = sparse.coo_matrix((datafile['D_values'], 
                                (datafile['D_indices'][0,:], datafile['D_indices'][1,:])), 
                                shape=datafile['D_shape'])
    D_csr = D_coo.tocsr()
    D_csr_chunk = D_csr[:, (rank*p_chunk_size):((rank+1)*p_chunk_size)]
    D_coo_chunk = D_csr_chunk.tocoo()
    D_values = TType(D_coo_chunk.data)
    D_rows   = torch.LongTensor(D_coo_chunk.row)
    D_cols   = torch.LongTensor(D_coo_chunk.col)
    if D_values.is_cuda:
        D_rows = D_rows.cuda()
        D_cols = D_cols.cuda()
    D_indices = torch.stack([D_rows, D_cols], dim=1).t()
    D_shape = D_coo_chunk.shape
    D_size  = torch.Size([int(D_shape[0]), int(D_shape[1])])
    D_chunk = TType_sp(D_indices, D_values, D_size).t()
    D_dist  = THDistMat.from_chunks(D_chunk).t()

    counts = TType(datafile['counts']) # put everywhere

    pet = PET(counts, e_dist, G_dist, D_dist, mu=float(args.mu), TType=TType)
    pet.run(check_obj=False, tol=float(args.tol), check_interval=100, maxiter=int(args.iter))
