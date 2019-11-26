import torch
import torch.distributed as dist
from . import distmat
from .distmat import THDistMat
import time
def euclidean_distance_tensor(dataA, dataB, out=None, splits=1):
    v1  = torch.sum(dataA**2, 1).view(-1,  1)
    v2  = torch.sum(dataB**2, 1).view( 1, -1)
    ABt = torch.mm(dataA, torch.t(dataB), out=out)
    rsq = torch.clamp(v1+v2-2*ABt, min=0, out=out)
    #print(rsq.shape)
    r = rsq
    assert rsq.shape[0] % splits == 0
    sz = rsq.shape[0] // splits
    for i in range(splits):
        r_part = r[i*sz:(i+1)*sz, :]
        torch.sqrt_(r_part) 

    #print(torch.isnan(r).any())
    #print((r<0).any())
    #print((r<=0).any())
    #print((r<=0).all())

    return r
    """
    if dataA is dataB:
        AAt = torch.mm(dataA, torch.t(dataA), out=out)
        d   = torch.diag(AAt)
        v1  = d.view(-1, 1)
        v2  = d.view( 1,-1)
        rsq = v1 + v2 - 2*AAt
        q   = rsq.shape[0]
        return torch.sqrt(rsq, out=out)
    else:
        v1  = torch.sum(dataA**2, 1).view(-1,  1)
        v2  = torch.sum(dataB**2, 1).view( 1, -1)
        ABt = torch.mm(dataA, torch.t(dataB), out=out)
        return torch.sqrt(v1+v2-2*ABt, out=out)
    """
def euclidean_distance_DistMat(data, verbose=False):
    rank = dist.get_rank()
    size = dist.get_world_size()

    assert data.byrow

    out_chunk = data.chunk.new(data.sizes[rank], data.shape[0])
    out_split = torch.split(out_chunk, data.sizes, dim=1)

    for i in range(size):
        if verbose: 
            print("Step {} of {}...".format(i+1, size))
        t0 = time.time()
        this  = data.chunk
        if i == rank:
            other = data.chunk
        else:
            other = data.chunk.new(data.sizes[i], data.shape[1])
        dist.broadcast(other, i)
        euclidean_distance_tensor(this, other, out=out_split[i])
        t1 = time.time()
        if verbose:
            print("Done step {}, time elapsed:{}".format(i+1, t1-t0))

    return THDistMat.from_chunks(out_chunk)









