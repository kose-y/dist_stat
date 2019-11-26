import os
import torch
import torch.distributed as dist
import numpy as np
import math
from .utils import coo_to_sparsetensor
from . import partitioners
size_from_split = lambda split, rank: split[rank+1]-split[rank]
synchronize = lambda : torch.cuda.synchronize() if torch.cuda._initialized else None

class THDistMat():
    """
    Distributed Matrix. 1D partitioning. 
    """
    def __init__(self, shape, sizes, chunk, byrow=True):
        self.shape = shape
        if not isinstance(self.shape, list):
            self.shape = list(shape)
        if torch.is_tensor(self.shape[0]):
            self.shape[0] = self.shape[0].item()
        if torch.is_tensor(self.shape[1]):
            self.shape[1] = self.shape[1].item()
        self.sizes = sizes
        if not isinstance(self.sizes, list):
            self.sizes = list(sizes) 
        if torch.is_tensor(self.sizes[0]):
            self.sizes = [x.item() for x in self.sizes]
        # check if sum of sizes is original shape
        if byrow:
            assert sum(sizes) == self.shape[0]
        else:
            assert sum(sizes) == self.shape[1]

        self.chunk = chunk
        self.byrow = byrow
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        dist.barrier()

    def coalesce(self):
        assert self.chunk.layout == torch.sparse_coo
        self.chunk = self.chunk.coalesce()
        #print(self.chunk.shape)
        return self


    def apply(self, op, *args, **kwargs): # apply operation. number of rows should not change.
        rank = dist.get_rank()
        chunk = op(self.chunk, *args, **kwargs)
        #assert self.sizes[rank] == chunk.shape[0 if self.byrow else 1]
        shape = [self.shape[0], chunk.shape[1]] if self.byrow else [chunk.shape[0], self.shape[1]]
        return THDistMat(shape, self.sizes, chunk, self.byrow)
    def apply_binary(self, other, op, *args, **kwargs):# apply binary operation. number of rows should not change.
        #rank = dist.get_rank()
        if isinstance(other, THDistMat):
            if self.byrow == other.byrow:
                otherchunk = other.chunk
            else: # "outer" operation. all_gather other.to outerchunk.
                otherchunk = other.chunk.new_zeros(other.shape[0], other.shape[1])
                #print(otherchunk)
                dim = 0 if other.byrow else 1
                #sizes = [other.chunk.shape[dim] for _ in range(other.size)]
                otherchunk_thin = list(torch.split(otherchunk, other.sizes, dim=dim))
                dist.barrier()
                synchronize()
                dist.all_gather(otherchunk_thin, other.chunk)
                dist.barrier()
                synchronize()
                #print(otherchunk[0:10], self.rank)
                #print(otherchunk[5000:5010], self.rank) 

        else:
            otherchunk = other
        chunk = op(self.chunk, otherchunk, *args, **kwargs)
        #assert self.sizes[rank] == chunk.shape[0 if self.byrow else 1]
        shape = [self.shape[0], chunk.shape[1]] if self.byrow else [chunk.shape[0], self.shape[1]]
        dist.barrier()
        return THDistMat(shape, self.sizes, chunk, self.byrow)
        
    def apply_inplace(self, op, *args, **kwargs):
        self.chunk = op(self.chunk, *args, **kwargs)
        return self
    def apply_binary_inplace(self, other, op, *args, **kwargs):
        if isinstance(other, THDistMat):
            otherchunk = other.chunk
        else:
            otherchunk = other
        self.chunk = op(self.chunk, otherchunk, *args, **kwargs)
        return self
    def type(self, t):
        chunk = chunk.type(t)
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)
    def t(self): # transpose
        return THDistMat(torch.LongTensor([self.shape[1], self.shape[0]]), self.sizes, 
                            torch.t(self.chunk), not self.byrow)
    def __neg__(self):
        return THDistMat(self.shape, self.sizes, -self.chunk, self.byrow)
    def __add__(self, other):
        return self.apply_binary(other, lambda x,y: x+y)
    def __mul__(self, other):
        return self.apply_binary(other, lambda x,y: x*y)
    def __sub__(self, other):
        return self.apply_binary(other, lambda x,y: x-y)
    def __truediv__(self, other):
        return self.apply_binary(other, lambda x,y: x/y)
    def __pow__(self, other):
        return self.apply_binary(other, lambda x,y: x**y)
    def __radd__(self, other):
        return self.apply_binary(other, lambda x,y: y+x)
    def __rmul__(self, other):
        return self.apply_binary(other, lambda x,y: y*x)
    def __rsub__(self, other):
        return self.apply_binary(other, lambda x,y: y-x)
    def __rtruediv__(self, other):
        return self.apply_binary(other, lambda x,y: y/x)
    def __lt__(self, other):
        return self.apply_binary(other, lambda x,y: x<y)
    def __le__(self, other):
        return self.apply_binary(other, lambda x,y: x<=y)
    def __eq__(self, other):
        return self.apply_binary(other, lambda x,y: x==y)
    def __ne__(self, other):
        return self.apply_binary(other, lambda x,y: x!=y)
    def __ge__(self, other):
        return self.apply_binary(other, lambda x,y: x>=y)
    def __gt__(self, other):
        return self.apply_binary(other, lambda x,y: x>y)
    def __iadd__(self, other):
        return self.add_(other)
    def __isub__(self, other):
        return self.sub_(other)
    def __imul__(self, other):
        return self.mul_(other)
    def __itruediv__(self, other):
        return self.div_(other)

    # some transformations
    def sqrt(self):
        chunk = self.chunk.sqrt()
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)
    def abs(self):
        chunk = self.chunk.abs()
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)
    def exp(self):
        chunk = self.chunk.exp()
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)
    def log(self):
        chunk = self.chunk.log()
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)
    def cumsum(self, dim):
        new_chunk = self.chunk.cumsum(dim)
        if self.byrow and dim==0:
            buf = torch.zeros_like(new_chunk[-1, :])
            for i in range(self.size-1):
                if self.rank == i: 
                    synchronize()
                    dist.send(new_chunk[-1,:], i+1)
                elif self.rank == i + 1:
                    synchronize()
                    dist.recv(buf, i)
                    new_chunk += buf
        elif not self.byrow and dim==1:
            buf = torch.zeros_like(new_chunk[:, -1])
            for i in range(self.size-1):
                if self.rank==i:
                    synchronize()
                    dist.send(new_chunk[:, -1], i+1)
                elif self.rank == i+1:
                    synchronize()
                    dist.recv(buf, i)
                    new_chunk += buf
        return THDistMat(self.shape, self.sizes, new_chunk, self.byrow)
        

      
        

    # some in-place methods
    def zero_(self):
        return self.apply_inplace(lambda x: x.zero_())
    def add_(self, other):
        if isinstance(other, THDistMat):
            self.chunk.add_(other.chunk)
        else:
            self.chunk.add_(other)
        return self
    def div_(self, other):
        #return self.apply_binary_inplace(other, lambda x,y: x.div_(y)) # this is equivalent, but slower
        if isinstance(other, THDistMat):
            self.chunk.div_(other.chunk)
        else:
            self.chunk.div_(other)
        return self
        
    def sub_(self, other):
        #return self.apply_binary_inplace(other, lambda x,y: x.sub_(y))
        #assert self.shape == other.shape
        #assert self.sizes == other.sizes
        if isinstance(other, THDistMat):
            self.chunk.sub_(other.chunk)
        else:
            self.chunk.sub_(other)
        return self
        
    def mul_(self, other):
        #return self.apply_binary_inplace(other, lambda x,y: x.mul_(y))
        #assert self.shape == other.shape
        #assert self.sizes == other.sizes
        if isinstance(other, THDistMat):
            self.chunk.mul_(other.chunk)
        else:
            self.chunk.mul_(other)
        return self
    def pow_(self, exponent):
        #return self.apply_inplace(lambda x: x.pow_(exponent))
        self.chunk.pow_(exponent)
        return self
    def abs_(self):
        self.chunk.abs_()
        return self

    def copy_(self, other):
        self.chunk.copy_(other.chunk)
        return self

    # additional opeartions
    def all_reduce_sum(self):
        partial_sum = torch.sum(self.chunk)
        sum_tensor = self.chunk.new([partial_sum])
        synchronize()
        dist.all_reduce(sum_tensor, dist.reduce_op.SUM)
        return sum_tensor[0]
    def all_reduce_max(self):
        partial_max = torch.max(self.chunk)
        max_tensor = self.chunk.new([partial_max])
        synchronize()
        dist.all_reduce(max_tensor, dist.reduce_op.MAX)
        return max_tensor[0]
    def all_reduce_min(self):
        partial_min = torch.min(self.chunk)
        min_tensor = self.chunk.new([partial_min])
        synchronize()
        dist.all_reduce(min_tensor, dist.reduce_op.MIN)
        return min_tensor[0]
    def diag(self, distribute=True):
        """
        get diagonal. 
        distribute: True to get the diagonal as a distributed matrix.
              False to get the diagonal as a broadcasted vector via all_gather.
        """
        assert self.shape[0]==self.shape[1]
        rank = dist.get_rank()
        partition = torch.cumsum(torch.LongTensor([0] + self.sizes), 0)
        chunk = torch.diag(self.chunk, partition[rank].item()).view(-1, 1)
        if distribute:
            shape = [self.shape[0], 1]
            sizes = self.sizes
            byrow = True
            return THDistMat(shape, sizes, chunk, byrow)
        else:
            out       = self.chunk.new(partition[-1].item(), 1)
            out_split = list(torch.split(out, self.sizes, 0))
            synchronize()
            dist.all_gather(out_split, chunk)
            return out
    def fill_diag_(self, value):
        '''
        fill diagonal in-place 
        '''
        assert self.shape[0] == self.shape[1]
        rank = dist.get_rank()
        p = self.sizes[rank]
        q = self.shape[0]
        partition = torch.cumsum(torch.LongTensor([0] + self.sizes), 0)
        diag      = self.chunk.view(-1)[(partition[rank].item())::(q+1)] 
        diag.fill_(value)
        return self
    def type(self, t):
        chunk = self.chunk.type(t)
        return THDistMat(self.shape, self.sizes, chunk, self.byrow)

    def sum(self, dim=None):
        if dim is None:
            return self.all_reduce_sum()
        else:
            if self.byrow:
                if dim==0:
                    partial = torch.sum(self.chunk, dim=0).view(1,-1)
                    synchronize()
                    dist.all_reduce(partial, dist.reduce_op.SUM)
                    return partial
                     
                elif dim==1:
                    chunk = torch.sum(self.chunk, dim=1).view(-1, 1)
                    return THDistMat.from_chunks(chunk)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")
            else:
                if dim==1:
                    partial = torch.sum(self.chunk, dim=1, keepdim=True)
                    synchronize()
                    dist.all_reduce(partial, dist.reduce_op.SUM)
                    return partial
                elif dim==0:
                    chunk = torch.sum(self.chunk, dim=0, keepdim=True)
                    return THDistMat.from_chunks(chunk, force_bycol=True)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")
    def max(self, dim=None):
        if dim is None:
            return self.all_reduce_max()
        else:
            if self.byrow:
                if dim==0:
                    partial = torch.max(self.chunk, dim=0, keepdim=True)
                    synchronize()
                    return dist.all_reduce(partial, dist.reduce_op.MAX)
                elif dim==1:
                    chunk = torch.max(self.chunk, dim=1, keepdim=True)
                    return THDistMat.from_chunks(chunk)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")
            else:
                if dim==1:
                    partial = torch.max(self.chunk, dim=1, keepdim=True)
                    synchronize()
                    return dist.all_reduce(partial, dist.reduce_op.MAX)
                elif dim==0:
                    chunk = torch.max(self.chunk, dim=0, keepdim=True)
                    return THDistMat.from_chunks(chunk, force_bycol=True)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")
    def min(self, dim=None):
        if dim is None:
            return self.all_reduce_min()
        else:
            if self.byrow:
                if dim==0:
                    partial = torch.min(self.chunk, dim=0, keepdim=True)
                    synchronize()
                    return dist.all_reduce(partial, dist.reduce_op.MIN)
                elif dim==1:
                    chunk = torch.min(self.chunk, dim=1, keepdim=True)
                    return THDistMat.from_chunks(chunk)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")
            else:
                if dim==1:
                    partial = torch.min(self.chunk, dim=1, keepdim=True)
                    synchronize()
                    return dist.all_reduce(partial, dist.reduce_op.MIN)
                elif dim==0:
                    chunk = torch.min(self.chunk, dim=0, keepdim=True)
                    return THDistMat.from_chunks(chunk, force_bycol=True)
                else:
                    raise NotImplementedError("dim can be None, 0, or 1.")

    @classmethod
    def from_chunks(cls, chunk, force_bycol=False):
        rank = dist.get_rank()
        size = dist.get_world_size()
        byrow = True
        if chunk.is_sparse:
            byrow=True
        else:
            try:
                chunk.view(-1)
            except:
                byrow=False
        if force_bycol:
            byrow=False
        
        size_member_tensor = torch.LongTensor([chunk.shape[0 if byrow else 1]])
        sizes_tensor = torch.LongTensor(size)
        size_tensor_list = list(torch.split(sizes_tensor, 1))
        dist.all_gather(size_tensor_list, size_member_tensor)
        shape = ([torch.sum(sizes_tensor),chunk.shape[1]] if byrow else 
                    [chunk.shape[0], torch.sum(sizes_tensor)])
        sizes = list(sizes_tensor)
        return THDistMat(shape, sizes, chunk, byrow)
    @classmethod
    def from_spmatrix(cls, D, partitioner=partitioners.default_partitioner, TType=torch.sparse.DoubleTensor):
        rank = dist.get_rank()
        size = dist.get_world_size()
        D_csr = D.tocsr()
        n_r = D.shape[0]
        nd_r = size
        
        partition_r = partitioner_r(n_r, nd_r)

        chunk_csr = D_csr[partition_r[rank]:partition_r[rank+1]]
        chunk_tensor = coo_to_sparsetensor(D_csr.tocoo(), TType=TType)
        return from_chunks(chunk_tensor)

    @classmethod
    def mm(cls, distmatA, distmatB): #TODO
        pass


#@profile        
def t(mat):
    if isinstance(mat, THDistMat):
        return mat.t()
    elif torch.is_tensor(mat):
        return torch.t(mat)
    elif mat is None:
        return None
    else:
        raise NotImplementedError("one of THDistMat, Tensor, and None is accepted")
#@profile
def diag(mat, distribute=True):
    if isinstance(mat, THDistMat):
        return mat.diag(distribute)
    elif torch.is_tensor(mat):
        return torch.diag(mat)
    elif mat is None:
        return None
    else:
        raise NotImplementedError("one of THDistMat, Tensor, and None is accepted")
#def sum(mat, dim=None):
#    if isinstance(mat, THDistMat):
#        return mat.sum(dim=dim)
#    elif torch.is_tensor(mat):
#        return torch.sum(mat, dim=dim, keepdim=True)
#    elif mat is None:
#        return None
#    else:
#        return sum(mat)
#@profile
def add(a, b, out=None):
    if not isinstance(a, THDistMat) and isinstance(b, THDistMat):
        return b.__radd__(a)
    else:
        return a+b
    #rank = dist.get_rank()
    #if out is None:
    #    outchunk = a.chunk.new(a.chunk.shape[0], a.chunk.shape[1])
    #    out = THDistMat.from_chunks(outchunk)
    #torch.add(a.chunk,  other=b.chunk, out=out.chunk)
    #return out
#@profile
def sub(a, b, out=None):
    if not isinstance(a, THDistMat) and isinstance(b, THDistMat):
        return b.__rsub__(a)
    else:
        return a-b
    #rank = dist.get_rank()
    #if out is None:
    #    outchunk = a.chunk.new(a.chunk.shape[0], a.chunk.shape[1])
    #    out = THDistMat.from_chunks(outchunk)
    #torch.add(a.chunk, value=-1.0, other=b.chunk, out=out.chunk)
    #return out
#@profile
def mul(a, b, out=None):
    if not isinstance(a, THDistMat) and isinstance(b, THDistMat):
        return b.__rmul__(a)
    else:
        return a*b
#@profile
def div(a, b, out=None):
    if not isinstance(a, THDistMat) and isinstance(b, THDistMat):
        return b.__rdiv__(a)
    else:
        return a/b
#@profile
def abs(a, out=None):
    if out is None:
        outchunk = a.chunk.new(a.chunk.shape)
        out = THDistMat.from_chunks(outchunk)
    torch.abs(a.chunk, out=out.chunk)
    return out
#@profile
def sign(a, out=None):
    if out is None:
        outchunk = a.chunk.new(a.chunk.shape)
        out = THDistMat.from_chunks(outchunk)
    torch.sign(a.chunk, out=out.chunk)
    return out
#@profile
def square_diff(a, b, weight=None, tmp=None):
    diff = sub(a, b, out=tmp).pow_(2)
    if weight is not None:
        diff = diff.mul_(weight)
    return diff.all_reduce_sum()
#@profile
def l2_diff(a, b, weight=None, tmp=None):
    diff = sub(a, b, out=tmp).pow_(2)
    if weight is not None:
        diff = diff.mul_(weight)
    return math.sqrt(diff.all_reduce_sum())
#@profile
def l1_diff(a, b, tmp=None):
    diff = sub(a, b, out=tmp)
    diff.abs_()
    return diff.all_reduce_sum()
#@profile
def linf_diff(a, b, tmp=None):
    diff = sub(a, b, out=tmp)
    diff.abs_()
    return diff.all_reduce_max()
#@profile
def dist_data(data, src=0, TType=torch.DoubleTensor):
    '''
    distribute a row-major matrix
    '''
    rank = dist.get_rank()
    size = dist.get_world_size()

    if rank==src:
        p = data.shape[0]
        q = data.shape[1]
        shape = torch.LongTensor([p,q])
        if p%size != 0:
            sizes = [ p//size+1 for i in range(size-1) ] + [p//size+1-(size-p%size)]
        else:
            sizes = [p//size for i in range(size)]
        p, q = shape[0], shape[1]
        sizes = torch.LongTensor(sizes)
        dist.broadcast(shape, src)
        dist.broadcast(sizes, src)
    else:
        shape = torch.LongTensor(2)
        sizes = torch.LongTensor(size)
        dist.broadcast(shape, src)
        dist.broadcast(sizes, src)
        shape = list(shape)
        p,q = shape[0], shape[1]

    p_chunk = sizes[rank].item()
    q_chunk = q.item()
    # print(rank, p_chunk, q_chunk)
    chunk = TType(p_chunk, q_chunk)

    reqs = []
    if rank==src:
        data_ = TType(data.shape).copy_(data)
        sizes_int = tuple(x.item() for x in tuple(sizes))
        data_split = torch.split(data_, sizes_int)

        chunk.copy_(data_split[src])
        for i in range(size):
            if i == src: continue
            synchronize()
            reqs.append(dist.isend(data_split[i], i))
    else:
        synchronize()
        reqs.append(dist.irecv(chunk, src))

    for req in reqs:
        req.wait()

    dist.barrier()
    return THDistMat(shape, sizes, chunk, True)
#@profile
def distgen_base(p, q, byrow=True, TType=torch.DoubleTensor):
    '''
    generation of a distributed matrix
    output: (p) x q matrix (p x (q) if byrow is False)
    '''
    if torch.is_tensor(p):
        p = p.item()
    if torch.is_tensor(q):
        q = q.item()
    rank = dist.get_rank()
    size = dist.get_world_size()

    if byrow:
        if p%size != 0:
            sizes = [ p//size+1 for i in range(size-1) ] + [p//size+1-(size-p%size)]
        else:
            sizes = [p//size for i in range(size)]
        p_chunk = sizes[rank]
        q_chunk = q
    else:
        if q%size != 0:
            sizes = [ q//size+1 for i in range(size-1) ] + [q//size+1-(size-q%size)]
        else:
            sizes = [q//size for i in range(size)]
        p_chunk = p
        q_chunk = sizes[rank]
        
    shape = torch.LongTensor([p, q])
    if byrow:
        chunk = TType(p_chunk, q_chunk)
    else:
        chunk = torch.t(TType(q_chunk, p_chunk))
    byrow = byrow
    return THDistMat(shape,sizes, chunk, byrow)
#@profile
def distgen_normal(p, q, byrow=True, TType=torch.DoubleTensor, set_from_master=True):
    '''
    distributed generation of normal random data
    output: (p) x q matrix (p x (q) if byrow is False)
    set_from_master: when you need to use the same initialization over multiple settings. 
    '''
    if torch.is_tensor(p):
        p = p.item()
    if torch.is_tensor(q):
        q = q.item()
    if not set_from_master:
        m = distgen_base(p, q, byrow, TType)
        m.chunk.normal_()
        return m
    else:
        if byrow:
            r = torch.DoubleTensor(p, q).normal_().type(TType)
            return dist_data(r, TType=TType)
        else:
            r = torch.DoubleTensor(q, p).normal_().type(TType)
            return dist_data(r, TType=TType).t()
        
#@profile
def distgen_uniform(p, q, lo=0, hi=1, byrow=True, TType=torch.DoubleTensor, set_from_master=True):
    '''
    distributed generation of uniform random data
    output: (p) x q matrix (p x (q) if byrow is False)
    set_from_master: when you need to use the same initialization over multiple settings. 
    '''
    if torch.is_tensor(p):
        p = p.item()
    if torch.is_tensor(q):
        q = q.item()
    if not set_from_master:
        m = distgen_base(p, q, byrow, TType)
        m.chunk.uniform_(lo, hi)
        return m
    else:
        if byrow:
            r = torch.DoubleTensor(p,q).uniform_(lo, hi).type(TType)
            return dist_data(r, TType=TType)
        else:
            r = torch.DoubleTensor(q, p).uniform_(lo, hi).type(TType)
            return dist_data(r, TType=TType).t()

#@profile
def distgen_zero(p, q, byrow=True, TType=torch.DoubleTensor):
    '''
    distributed generation of zero data
    output: (p) x q matrix (p x (q) if byrow is False)
    '''
    if torch.is_tensor(p):
        p = p.item()
    if torch.is_tensor(q):
        q = q.item()
    m = distgen_base(p, q, byrow, TType)
    m.chunk.zero_()
    return m
def distgen_ones(p, q, byrow=True, TType=torch.DoubleTensor):
    '''
    distributed generation of zero data
    output: (p) x q matrix (p x (q) if byrow is False)
    '''
    if torch.is_tensor(p):
        p = p.item()
    if torch.is_tensor(q):
        q = q.item()
    m = distgen_base(p, q, byrow, TType)
    m.chunk.fill_(1)
    return m

from .distmm import _distmm_thinfat_byrow, distmm_thinthin_inner, _distmm_fatthin_byrow, distmm_db_b, distmm_db_d, distmm_thinthin_outer

def mm(matA, matB, out_sizes=None, out=None, tmpout=None, bycol=None):
    if isinstance(matA, THDistMat) and isinstance(matB, THDistMat):
        if matA.byrow and not matB.byrow:
            if not bycol:
                        
                return   distmm_thinthin_outer(  matA ,   matB , tmpout=tmpout, out=out)
            else:
                return t(distmm_thinthin_outer(t(matB), t(matA), tmpout=t(tmpout), out=t(out)))
        elif matA.byrow and matB.byrow:
            return   _distmm_fatthin_byrow(  matA ,   matB , tmpout=tmpout, out=out)
        elif not matA.byrow and not matB.byrow:
            return t(_distmm_fatthin_byrow(t(matB), t(matA), tmpout=t(tmpout), out=t(out)))
        else: # not matA.byrow and matB.byrow
            if out_sizes is None:
                return distmm_thinthin_inner(matA, matB, out=out)
            else:
                if bycol is None:
                    # figure it out
                    out_dist_dim = sum(out_sizes)
                    if matA.shape[0] == out_dist_dim:
                        return t(_distmm_thinfat_byrow(t(matB), t(matA), out_sizes=out_sizes, tmpout=t(tmpout)))
                    elif matB.shape[1] == out_dist_dim:
                        return   _distmm_thinfat_byrow(  matA ,   matB , out_sizes=out_sizes, tmpout=tmpout)
                    else:
                        raise ValueError("sum of out_sizes does not match any of out dimensions")
                        
                elif not bycol:
                    return t(_ditsmm_thinfat_byrow(t(matB), t(matA), out_sizes=out_sizes, tmpout=t(tmpout)))
                else:
                    return   _distmm_thinfat_byrow(  matA ,   matB , out_sizes=out_sizes, tmpout=tmpout)
    elif isinstance(matA, THDistMat) and not isinstance(matB, THDistMat):
        if matA.byrow:
            return   distmm_db_d(  matA ,   matB , out=out)
        else:
            return   distmm_db_b(  matA ,   matB , out=out)
    elif not isinstance(matA, THDistMat) and isinstance(matB, THDistMat):
        if not matB.byrow:
            return t(distmm_db_d(t(matB), t(matA), out=t(out))) # distributed comes first
        else:
            return t(distmm_db_b(t(matB), t(matA), out=t(out))) # distributed comes first
    else:
        return torch.mm(matA, matB)

