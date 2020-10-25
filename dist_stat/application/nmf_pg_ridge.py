import torch
import torch.distributed as dist
from .. import distmat
from ..distmat import THDistMat, distgen_uniform, distgen_base, linf_diff, l2_diff
from ..distmm import distmm_fatthin, distmm_thinthin_inner, distmm_db_d, distmm_thinthin_outer
import time
from math import inf
"""
NMF, without intermediate storages
"""

class NMF():
    def __init__(self, data, r, eps=1e-3, TType=torch.DoubleTensor, init_from_master=False):
        """
        data: a THDistMat, byrow.
        r: a small integer
        """
        assert data.byrow
        self.TType = TType

        self.p, self.q = data.shape
        self.r = r
        p, q = self.p, self.q
        assert self.r<=self.p and self.r<=self.q

        # initialize V and W
        self.V_prev = distgen_base(p, r, TType=TType)
        self.W_prev = distgen_base(q, r, TType=TType).t()

        self.V = distgen_uniform(p, r, TType=TType, set_from_master=init_from_master)
        self.W = distgen_uniform(q, r, TType=TType, set_from_master=init_from_master).t()

        self.data = data
        #self.data_double = data.type(torch.DoubleTensor)
        self.prev_obj = inf
        self.eps=eps

    def update_V(self):
        Wt = self.W.t()
        XWt =  distmat.mm(self.data, Wt)
        WWt =  distmat.mm(self.W, Wt)
        VWWt = distmat.mm(self.V, WWt)
        sigma_k = 1.0/(2*((WWt**2).sum() + self.eps * self.r).sqrt())
        self.V = (self.V * (1.0 - sigma_k * self.eps) - (VWWt - XWt)*sigma_k).apply(torch.clamp, min=0)
    def update_W(self):
        VtX  = distmat.mm(self.V.t(), self.data, out_sizes=self.W.sizes) 
        VtV  = distmat.mm(self.V.t(), self.V)
        VtVW = distmat.mm(VtV, self.W)
        tau_k = 1.0/(2*((VtV**2).sum() + self.eps * self.r).sqrt())
        self.W = (self.W * (1.0 - tau_k * self.eps) - (VtVW-VtX)*tau_k).apply(torch.clamp, min=0)
    def get_objective(self):
        outer = distmat.mm(self.V, self.W)
        #val =  l2_diff(self.data, outer)**2
        val = ((self.data - outer)**2).all_reduce_sum() + self.eps * ((self.V**2).all_reduce_sum() + (self.W**2).all_reduce_sum())
        return val
    def check_convergence(self,tol, verbose=True, check_obj=False, check_interval=1):
        rank = dist.get_rank()
        obj = None
        diff_norm_1 = linf_diff(self.V_prev, self.V)
        diff_norm_2 = linf_diff(self.W_prev, self.W)
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm_1 < tol and diff_norm_2 < tol
        return (diff_norm_1, diff_norm_2, reldiff, obj), converged
    def run(self, maxiter=100, tol=1e-5, check_interval=1, verbose=True, check_obj=False):
        rank = dist.get_rank()
        if verbose:
            if rank==0:
                print("Starting...")
                print("p={}, q={}, r={}".format(self.p, self.q, self.r))
                if not check_obj:
                    print("%6s\t%15s\t%15s\t%10s" % ("iter", "V_maxdiff", "W_maxdiff", "time"))
                else:
                    print("%6s\t%15s\t%15s\t%15s\t%15s\t%10s" % ("iter", "V_maxdiff", "W_maxdiff", "reldiff",  "obj", "time" ))
                print('-'*80)

        t0 = time.time()
        t_start = t0

        for i in range(maxiter):
            self.V_prev.copy_(self.V)
            self.W_prev.copy_(self.W)
            
            self.update_V()
            self.update_W()
            if (i+1) % check_interval ==0:
                t1 = time.time()
                (v_maxdiff, w_maxdiff, reldiff, obj), converged = self.check_convergence(tol, verbose, check_obj, check_interval=1)
                if verbose:
                    if not check_obj:
                        if rank ==0:
                            print("%6d\t%15.9e\t%15.9e\t%10.5f" % (i+1, v_maxdiff, w_maxdiff, t1-t0))
                    else:
                        if rank==0:
                            print("%6d\t%15.9e\t%15.9e\t%15.9e\t%15.9e\t%10.5f" % (i+1, v_maxdiff, w_maxdiff, reldiff,
                                                                            obj, t1-t0))
                if converged: break
                t0 = t1

            dist.barrier()
        if verbose:
            if rank==0:
                print('-'*80) 
                print("Completed. total time: {}".format(time.time()-t_start))
        return self.V, self.W
