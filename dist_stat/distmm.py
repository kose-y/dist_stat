import os
import torch
import torch.distributed as dist
from .distmat import THDistMat, synchronize
# Consider NMF with: X ((p) x q and p x (q); two copies), V ((p) x r), W (r x (q))

#TODO: use group for dist ops
# before that, we need to check rank in broadcast, reduce, etc. 
# is relative to group, or is of the world.

#@profile
def distmm_fatthin(fat, thin, reverse=False, out_sizes=None, tmpout=None, out=None):
    '''
    wrapper to the two functions below. 
    reverse is thin x fat, any ordering.
    '''
    if fat.byrow:
        if not reverse:
            return _distmm_fatthin_byrow(fat, thin, False, tmpout, out)
        else:
            assert out_sizes is not None
            return _distmm_thinfat_byrow(thin, fat, out_sizes, False, tmpout)
    else:
        if not reverse:
            assert out_sizes is not None
            return _distmm_thinfat_byrow(thin, fat, out_sizes, True, tmpout)
        else:
            return _distmm_fatthin_byrow(fat, thin, True, tmpout, out)
            

#@profile
def _distmm_thinfat_byrow(thin, fat, out_sizes, reverse=False, tmpout=None):
    '''
    A[thin] (r x (p)), B[fat] ((p) x q) => AB(r x (q)). sizes for q is required. 
    reverse is fat(col-major) x thin(row-major) => thin(row-major).
    tmpout: r x q col-major if not reverse
    piece of tmpout is used as output
    compute the partial results and then reduce
    in NMF: V^T X
    '''
    assert fat.byrow is not reverse
    assert thin.byrow is reverse
    rank = dist.get_rank()

    if not reverse:
        shape = torch.LongTensor([thin.shape[0], fat.shape[1]])
    else:
        shape = torch.LongTensor([fat.shape[0], thin.shape[1]])

    # compute 
    if tmpout is None:
        if not reverse: # col-major storage
            tmpout = torch.t(thin.chunk.new(fat.shape[1], thin.shape[0]))
        else: # row-major storage
            tmpout = thin.chunk.new(fat.shape[0], thin.shape[1])
    else:
        if not reverse:
            assert tmpout.shape == torch.Size([thin.shape[0], fat.shape[1]])
            torch.t(tmpout).view(-1)
        else:
            assert tmpout.shape == torch.Size([fat.shape[0], thin.shape[1]])
            tmpout.view(-1)
    if not reverse:
        torch.mm(thin.chunk, fat.chunk, out=tmpout)
    else:
        torch.mm(fat.chunk, thin.chunk, out=tmpout)

    # split and reduce
    split_tmpout = torch.split(tmpout, out_sizes, dim=(1 if not reverse else 0))

    #tmp = [torch.t(fat.chunk.new(tmpout.size()[0], out_sizes[0])) for i in range(len(out_sizes))]
    #print ([x.size() for x in tmp])
    for i in range(len(out_sizes)):
        #if rank==i:
        #    dist.gather(torch.t(split_tmpout[i]), dst=i, gather_list=tmp)
        #else:
        #    dist.gather(torch.t(split_tmpout[i]), dst=i)
        if not reverse:
            synchronize()
            dist.reduce(torch.t(split_tmpout[i]), i, dist.reduce_op.SUM)
        else:
            synchronize()
            dist.reduce(split_tmpout[i], i, dist.reduce_op.SUM)

    byrow = reverse
    return THDistMat(shape, out_sizes, split_tmpout[rank], byrow)

#@profile
def _distmm_fatthin_byrow(fat, thin, reverse=False, tmpout=None, out=None):
    '''
    A[fat] ((p) x q), B[thin] ((q) x r) => AB((p) x r)
    reverse is thin(col-major) x fat(col-major) =>  col-major.
    tmpout: q x r, row-major if not reverse
    out: [p] x r, row-major if not reverse
    all_gather B first, and computes the multiplication.
    in NMF: X W^T
    '''
    assert fat.byrow is not reverse
    assert thin.byrow is not reverse

    rank = dist.get_rank()
    if not reverse:
        shape = torch.LongTensor([fat.shape[0], thin.shape[1]])
    else:
        shape = torch.LongTensor([thin.shape[0], fat.shape[1]])
    sizes = fat.sizes

    # all_gather 
    if tmpout is None:
        # storage for thin 
        if not reverse: # thin is row-major
            tmpout = thin.chunk.new(thin.shape[0], thin.shape[1])
        else: # thin is col-major
            tmpout = torch.t(thin.chunk.new(thin.shape[1], thin.shape[0]))
    else:
        if not reverse:
            assert tmpout.size() == torch.Size([thin.shape[0], thin.shape[1]])
            tmpout.view(-1)
        else:
            assert tmpout.size() == torch.Size([thin.shape[1], thin.shape[0]])
            torch.t(tmpout).view(-1)

    split_thin = list(torch.split(tmpout, thin.sizes, dim=(0 if not reverse else 1)))
    #print(rank, thin.chunk)
    synchronize()
    dist.all_gather(split_thin, thin.chunk)

    # compute
    if out is None:
        # storage for output
        if not reverse: # out is row-major
            out = thin.chunk.new(sizes[rank], shape[1])
        else: # out is col-major
            out = torch.t(thin.chunk.new(sizes[rank], shape[0]))
    else:
        if not reverse:
            assert out.shape == torch.Size([sizes[rank], shape[1]])
            out.view(-1)
        else:
            assert out.shape == torch.Size([shape[0], sizes[rank]])
            torch.t(out).view(-1)

    if not reverse:
        chunk = torch.mm(fat.chunk, tmpout, out=out)
    else:
        chunk = torch.mm(tmpout, fat.chunk, out=out)
    byrow = not reverse
    return THDistMat(shape, sizes, chunk, byrow)
    
#@profile
def distmm_thinthin_inner(matA, matB, out=None):
    '''
    A (r x (p)), B ((p) x s) => AB(r x s), out row-major 
    AB is copied to all devices via all_reduce.
    in NMF: (V^T V) and (W W^T)
    '''
    assert (not matA.byrow) and (matB.byrow)
    AB_chunk = torch.mm(matA.chunk, matB.chunk, out=out)
    synchronize()
    dist.all_reduce(AB_chunk, dist.reduce_op.SUM) # in-place op. 
    return AB_chunk
#@profile
def distmm_thinthin_outer(matA, matB, tmpout=None, out=None):
    '''
    A ((p) x r), B (r x (q)) => AB((p) x q), out row-major
    tmpout: r x q, to all_gather
    out: (p) x q
    B is all_gathered.
    in NMF: to compute objective.
    '''
    rank = dist.get_rank()
    assert matA.byrow and (not matB.byrow)
    p = matA.shape[0]
    q = matB.shape[1]
    r = matA.shape[1]
    assert r == matB.shape[0]
    shape = [p,q]
    sizes = matA.sizes
    byrow = True

    # all_gather
    if tmpout is None:
        tmpout = torch.t(matB.chunk.new(q, r))
    else:
        assert tmpout.size() == torch.Size([r,q])
        torch.t(tmpout).view(-1)
    split_tmpout = list(torch.split(tmpout, matB.sizes, dim=1))
    #print(split_tmpout)
    synchronize()
    dist.all_gather(split_tmpout, matB.chunk)

    # compute
    if out is None:
        out = matA.chunk.new(matA.sizes[rank], q)
    else:
        assert out.size() == torch.Size([matA.sizes[rank], q])
        out.view(-1)

    chunk = torch.mm(matA.chunk, tmpout, out=out)

    return THDistMat(shape, sizes, chunk, byrow)
        


#@profile
def distmm_db_d(distributed, broadcasted, reverse=False, out=None):
    '''
    A ((p) x r), B (r x s) => AB((p) x s)
    reverse: B(s x r) x A(r x (p)) => BA(s x (p))
    B is already copied to all devices.
    in NMF: (V^T V) W (reversed) and V (W W^T) 
    '''
    assert reverse is not distributed.byrow
    rank = dist.get_rank()
    shape = (torch.LongTensor([distributed.shape[0], broadcasted.shape[1]]) if not reverse
                else torch.LongTensor([broadcasted.shape[0], distributed.shape[1]]))
    sizes = distributed.sizes
    if out is None:
        if not reverse:
            out = broadcasted.new(sizes[rank], broadcasted.shape[1])
        else:
            out = torch.t(broadcasted.new(sizes[rank], broadcasted.shape[0]))
    else: 
        if not reverse:
            assert out.shape == torch.Size([sizes[rank], broadcasted.shape[1]])
            out.view(-1)
        else:
            assert out.shape == torch.Size([broadcasted.shape[0], sizes[rank]])
            torch.t(out).view(-1)
    chunk = (torch.mm(distributed.chunk, broadcasted, out=out) if not reverse
            else torch.mm(broadcasted, distributed.chunk, out=out))
    byrow = not reverse
    return THDistMat(shape, sizes, chunk, byrow)
#@profile
def distmm_db_b(distributed, broadcasted, out=None):
    '''
    A (r x (p)), B(p x s) => AB(r x s)
    B is broadcasted, output is broadcasted. 
    Example: dense matrix x broadcasted long vector.
    '''
    assert not distributed.byrow
    rank = dist.get_rank()
    shape = [distributed.shape[0], broadcasted.shape[1]]
    partition = torch.cumsum(torch.LongTensor([0] + distributed.sizes), 0)
    AB_chunk = torch.mm(distributed.chunk, broadcasted[partition[rank]:partition[rank+1], :], out=out)
    AB_chunk = torch.mm(distributed.chunk, broadcasted[partition[rank]:partition[rank+1], :], out=out)
    synchronize()
    dist.all_reduce(AB_chunk, dist.reduce_op.SUM) # in-place op. 
    return AB_chunk
    


