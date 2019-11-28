import torch
import argparse
import os
from dist_stat.pet_utils import *
from dist_stat.pet import PET
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse

def coo_to_sparsetensor(spm, TType=torch.DoubleTensor):
    typename = torch.typename(TType).split('.')[-1]
    TType_cuda = TType.is_cuda
    densemodule = torch.cuda if TType_cuda else torch
    spmodule = torch.cuda.sparse if TType_cuda else torch.sparse
    TType_sp = getattr(spmodule, typename)
    i = densemodule.LongTensor(np.vstack([spm.row, spm.col]))
    v = TType(spm.data)
    return TType_sp(i, v, spm.shape)

import torch.distributed as dist
dist.init_process_group('mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from dist_stat import distmat
from dist_stat.distmat import THDistMat
num_gpu = torch.cuda.device_count()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PET with squared difference penalty")
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
    parser.add_argument('--data', dest='data', action='store', default='../data/pet_gen_v2_100_120.npz',
                        help='data file (.npz)')
    parser.add_argument('--dense', dest='dense', action='store_const', default=False, const=True, 
                        help='use dense data matrix')
    parser.add_argument('--iter', dest='iter', action='store', default=1000, 
                        help='max iter')
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

    rank = dist.get_rank()
    size = dist.get_world_size()



    datafile = np.load(args.data)


    n_x = datafile['n_x']
    n_t = datafile['n_t']

    TType_name = torch.typename(TType).split('.')[-1]
    TType_sp   = getattr(torch.sparse, TType_name)
    if args.with_gpu:
        TType_sp   = getattr(torch.cuda.sparse, TType_name)
    #print(TType_sp)

    #e = torch.Tensor(datafile['e']).type(TType)
    e_indices = datafile["e_indices"]
    e_values = datafile["e_values"]

    p = n_x**2
    d = n_t * (n_t - 1) // 2
    p_chunk_size = p//size

    e_coo = coo_matrix((e_values, (e_indices[0,:], e_indices[1,:])), shape=(d, p))
    e_csc = e_coo.tocsc()
    e_csc_chunk = e_csc[:, (rank*p_chunk_size):((rank+1)*p_chunk_size)]
    e_coo_chunk = e_csc_chunk.tocoo()
    e_values = TType(e_coo_chunk.data)
    e_rows = torch.LongTensor(e_coo_chunk.row)
    e_cols = torch.LongTensor(e_coo_chunk.col)
    if e_values.is_cuda:
        e_rows = e_rows.cuda()
        e_cols = e_cols.cuda()
    e_indices = torch.stack([e_rows, e_cols], dim=1).t()
    e_shape = e_coo_chunk.shape
    e_size = torch.Size([int(e_shape[0]), int(e_shape[1])])
    e_chunk = TType_sp(e_indices, e_values, e_size).t()

    if not args.dense:
        e_dist = THDistMat.from_chunks(e_chunk).t()
    else:
        e_chunk = e_chunk.to_dense().t()
        e_dist = THDistMat.from_chunks(e_chunk, force_bycol=True)


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
    pet.run(check_obj=True, tol=float(args.tol), check_interval=100, maxiter=int(args.iter))
