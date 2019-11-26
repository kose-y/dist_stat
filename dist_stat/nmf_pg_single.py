import torch
import time
from math import inf
"""
NMF, without intermediate storages
"""

class NMF():
    def __init__(self, data, r, TType=torch.DoubleTensor):
        """
        data: a Tensor.
        r: a small integer
        """
        self.TType = TType

        self.p, self.q = data.shape
        self.r = r
        p, q = self.p, self.q
        assert self.r<=self.p and self.r<=self.q

        # initialize V and W
        self.V = torch.DoubleTensor(p, r).uniform_().type(TType)
        self.W = torch.DoubleTensor(q, r).uniform_().t().type(TType)

        self.V_prev = TType(p, r)
        self.W_prev = TType(q, r).t() # such transpositions are intentional.


        self.data = data.type(TType)
        self.prev_obj = inf

    def update_V(self, eps=1e-6):
        Wt = self.W.t()
        WWt = torch.mm(self.W, Wt)
        sigma_k = 1.0/(2*torch.sqrt(torch.sum(WWt**2))+eps)
        XWt = torch.mm(self.data, Wt)
        VWWt = torch.mm(self.V, WWt)
        self.V = torch.clamp(self.V - sigma_k * (VWWt - XWt), min=0.0)
    
    def update_W(self, eps=1e-6):
        Vt = self.V.t()
        VtX = torch.mm(Vt, self.data)
        VtV = torch.mm(Vt, self.V)
        VtVW = torch.mm(VtV, self.W)
        tau_k = 1.0/(2*torch.sqrt(torch.sum(VtV**2))+eps)
        self.W = torch.clamp(self.W - tau_k * (VtVW - VtX), min=0.0)

    def get_objective(self):
        outer = torch.mm(self.V, self.W)
        val = torch.sum(((self.data - outer)**2))
        return val
    def check_convergence(self,tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm_1 = torch.max(torch.abs(self.V_prev-self.V))
        diff_norm_2 = torch.max(torch.abs(self.W_prev-self.W))
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
        if verbose:
            print("Starting...")
            print("p={}, q={}, r={}".format(self.p, self.q, self.r))
            if not check_obj:
                print("%6s\t%15s\t%15s\t%10s" % ("iter", "V_maxdiff", "W_maxdiff", "time"))
            else:
                print("%6s\t%15s\t%15s\t%15s\t%15s\t%10s" % ("iter", "V_maxdiff", "W_maxdiff", "reldiff", "obj", "time" ))
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
                (v_maxdiff, w_maxdiff, reldiff, obj), converged = self.check_convergence(tol, verbose, check_obj, check_interval=check_interval)
                if verbose:
                    if not check_obj:
                        print("%6d\t%15.9e\t%15.9e\t%10.5f" % (i+1, v_maxdiff, w_maxdiff, t1-t0))
                    else:
                        print("%6d\t%15.9e\t%15.9e\t%15.9e\t%15.9e\t%10.5f" % (i+1, v_maxdiff, w_maxdiff, reldiff,
                                                                            obj, t1-t0))
                if converged: break
                t0 = t1

        if verbose:
            print('-'*80) 
            print("Completed. total time: {}".format(time.time()-t_start))

