import os
import torch
import torch.distributed as dist
import platform
import argparse

dist.init_process_group('mpi')

from dist_stat import distmat
from dist_stat.distmm import *
num_gpu = torch.cuda.device_count()
rank = dist.get_rank()
size = dist.get_world_size()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Distributed matrix multiplication")
    parser.add_argument('--gpu', dest='with_gpu', action='store_const', const=True, default=False, 
                        help='whether to use gpu')
    parser.add_argument('--double', dest='double', action='store_const', const=True, default=False,
                        help='use this flag for double precision. otherwise single precision is used.')
    parser.add_argument('--offset', dest='offset', action='store', default=0,
                        help='gpu id offset')
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

    p = 4; q = 8; r = 2
    if rank==0:
        fat   = TType(p, q).normal_()
        thin1 = TType(q, r).normal_()
        thin2 = TType(p, r).normal_()

    else:
        fat, thin1, thin2 = None, TType(q,r), TType(p,r)
    dist.broadcast(thin1,0)
    dist.broadcast(thin2,0)

    fat_dist   = distmat.dist_data(fat, src=0, TType=TType)
    thin1_dist = distmat.dist_data(thin1, src=0, TType=TType)
    thin2_dist = distmat.dist_data(thin2, src=0, TType=TType)

    # test distmm_thinthin_inner 
    if rank==0: 
        print("distmm_thinthin_inner: thin1^T x thin1")
        thin1_thin1 = torch.mm(torch.t(thin1), thin1)
        print("thin1^T x thin1: ", thin1_thin1)

    thin1_thin1_bd = distmm_thinthin_inner(thin1_dist.t(), thin1_dist)
    print("rslt in rank %d: "%(rank,), thin1_thin1_bd)

    dist.barrier()


    if rank==0:
        print("distmm_db_d: thin2 x thin1_thin1")
        correct = torch.mm(thin2, thin1_thin1)
        print(correct)
    rslt_dist = distmm_db_d(thin2_dist, thin1_thin1_bd)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)


    dist.barrier()

    if rank==0:
        print("distmm_db_d (reverse): thin1_thin1 x thin2^T")
        correct = torch.mm(thin1_thin1, torch.t(thin2))
        print(correct)
    rslt_dist = distmm_db_d(thin2_dist.t(), thin1_thin1_bd, True)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    dist.barrier()

    if rank==0:
        print("_distmm_fatthin_byrow: fat x thin1")
        correct = torch.mm(fat, thin1)
        print(correct)
    rslt_dist = distmm_fatthin(fat_dist, thin1_dist)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    dist.barrier()

    if rank==0:
        print("_distmm_fatthin_byrow (reverse): thin1^T x fat^T")
        correct = torch.mm(torch.t(thin1), torch.t(fat))
        print(correct)
    rslt_dist = distmm_fatthin(fat_dist.t(), thin1_dist.t(), reverse=True)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    dist.barrier()

    if rank==0:
        print("_distmm_thinfat_byrow: thin2^T x fat" )
        # Note: this is reverse for distmm_fatthin, non-reverse for inner ftn
        correct = torch.mm(torch.t(thin2), fat)
        print(correct)
    rslt_dist = distmm_fatthin(fat_dist, thin2_dist.t(), reverse=True, 
                                out_sizes=thin1_dist.sizes)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    dist.barrier()

    if rank==0:
        print("_distmm_thinfat_byrow (reverse): fat^T x thin2" )
        # Note: this is reverse for distmm_fatthin, non-reverse for inner ftn
        correct = torch.mm(torch.t(fat), thin2)
        print(correct)
    rslt_dist = distmm_fatthin(fat_dist.t(), thin2_dist, reverse=False, 
                                out_sizes=thin1_dist.sizes)
    print("rslt in rand %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    if rank==0:
        print("distmm_thinthin_outer: thin1 x thin2^T" )
        correct = torch.mm(thin1, torch.t(thin2))
        print(correct)
    rslt_dist = distmm_thinthin_outer(thin1_dist, thin2_dist.t())
    print("rslt in rank %d: "%(rank,), rslt_dist.chunk)
    print(rslt_dist.byrow)

    if rank==0:
        print("distmm_db_b: thin1^T x thin1(dense)" )
        correct = torch.mm(torch.t(thin1), thin1)
        print(correct)
    rslt_dist = distmm_db_b(thin1_dist.t(), thin1)
    print("rslt in rank %d: "%(rank,), rslt_dist)


    dist.barrier()
    if rank==0:
        print("now we check distributed sparse matrices.")

    def to_sparse(x):
        """ converts dense tensor x to sparse format """
        x_typename = torch.typename(x).split('.')[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())
    
    thin1_sparse_chunk = to_sparse(thin1_dist.chunk)
    thin1_sparse_dist = THDistMat.from_chunks(thin1_sparse_chunk)
    thin2_sparse_chunk = to_sparse(thin2_dist.chunk)
    thin2_sparse_dist = THDistMat.from_chunks(thin2_sparse_chunk)
    print(thin1_sparse_chunk.shape)
    print(thin1_sparse_dist.t().shape)
    print(thin1_dist.shape)
    if rank==0:
        print("correct: ", torch.mm(thin1.t(), thin1))

    r =  distmat.mm(thin1_sparse_dist.t(), thin1_dist )
    print("rslt: ", r)
    
     




