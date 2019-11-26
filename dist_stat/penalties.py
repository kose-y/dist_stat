from scipy.sparse import csc_matrix
import numpy as np
import torch
from . import distmat
from .utils import gidx_to_partition
import torch.distributed as dist

class PenaltyFunction:
    def __init__(self, name=None):
        self.name = name or type(self).__name__
    def prox_(self, y): # in-place op
        raise NotImplementedError
    def eval(self, y):
        raise NotImplementedError
    def initialize(self):
        pass

class L1Penalty(PenaltyFunction):
    def __init__(self, lam, name=None):
        super().__init__(name)
        self.lam = lam
    def prox_(self, pre_prox, scale): 
        if isinstance(pre_prox, THDistMat):
            return pre_prox.apply_inplace(self.prox_, scale)
        return pre_prox
    def eval(self, Dx):
        return self.lam * Dx.abs().sum()

class GroupLasso(PenaltyFunction):
    def __init__(self, lam, g, partition=None, TType=torch.DoubleTensor):
        assert all([g[i+1] >= g[i] for i in range(len(g)-1)])
        self.lam = lam
        self.g = g.flatten()
        self.gpart = gidx_to_partition(self.g)
        self.TType = TType
        self.partition=partition
        super().__init__(name)

    def initialize(self):
        rank = dist.get_rank()
        size = dist.get_world_size()
        gpt = self.gpart
        sizes = np.array([gpt[i+1] - gpt[i] for i in range(len(gpt)-1)]).reshape((-1, 1))
        grpmat = csc_matrix((np.ones_like(self.g), self.g, np.arange(self.g.shape[0]+1))).tocsr().tocoo()
        sqrt_sizes = np.sqrt(sizes)
        if self.partition is None:
            self.grpmat = coo_to_sparsetensor(grpmat)
            self.sqrt_sizes = TType(sqrt_sizes)
            if self.sqrt_sizes.is_cuda:
                self.grpidx = torch.cuda.LongTensor(self.g)
            else:
                self.grpidx = torch.LongTensor(self.g)
            self.grpidx_2d = self.grpidx.view(-1,1)
            self.max_norms = self.lam * self.sqrt_sizes
            self.maxynorm  = torch.sqrt((self.max_norms**2).sum())
        else:
            partition = self.partition
            grp_device_partitioner = partitioners.groupvar_partitioner(partition, gpt)
            self.grp_device_part   = grp_device_partitioner(len(gpt)-1, dist.get_world_size())

            self.grpmat = THDistMat.from_spmatrix(grpmat, partitioner=grp_device_partitioner)


            self.sqrt_sizes = TType(sqrt_sizes[grp_device_part[rank]:grp_device_part[rank+1]])
            g_sect = self.g[partition[rank]:partition[rank+1]]
            g_sect = g_sect - np.min(g_sect)
            if self.sqrt_sizes.is_cuda:
                gidx   = torch.cuda.LongTensor(g_sect)
            else:
                gidx   = torch.LongTensor(g_sect)
            self.grpidx_2d = gidx.view(-1,1)
            self.max_norms = self.lam * self.sqrt_sizes
        self.sqrt_sizes = THDistMat.from_chunks(self.sqrt_sizes)
        self.grpidx_2d  = THDistMat.from_chunks(self.grpidx_2d)
        self.max_norms  = THDistMat.from_chunks(self.max_norms)
        self.maxynorm   = torch.sqrt((self.max_norms**2).sum())

    def prox_(self, pre_prox, scale):
        if self.partition is None:
            sumsq = torch.mm(self.grpmat, pre_prox**2)
            norms = torch.sqrt(sumsq)
            factors = self.max_norms/(torch.max(self.max_norms, norms))
            factors_elem = torch.gather(factors, 0, self.grpidx_2d)
            return pre_prox.mul_(factors_elem)
        else:
            suasq = distmat.mm(self.grpmat, pre_prox**2)
            norms = sumsq.sqrt()
            factors = self.max_norms.apply_binary(norms, lambda x,y: x/(torch.max(x,y)))
            factors_elem = factors.apply_binary(self.grpidx_2d, torch.gather, 0)
            return pre_prox.mul_(factors_elem)

    def eval(self, Dx):
        if self.partition is None:
            Dx_sumsq = torch.mm(self.grpmat, Dx**2)
            Dx_norms = torch.sqrt(Dx_sumsq)
            return self.lam * torch.mm(self.sqrt_sizes.t(), Dx_norms)[0]
        else:
            Dx_sumsq = distmat.mm(self.grpmat, Dx**2)
            Dx_norms = Dx_sumsq.sqrt()
            product = distmat.mm(self.sqrt_sizes.t(), Dx_norms)
            return (self.lam * product)[0] # check this



