import torch
import torch.distributed as dist
import time
from math import inf
from .. import distmat
import os
"""
l1-regularized cox regression
"""

class COX():
    def __init__(self, data, delta, lambd, seed=16962, sigma='power', TType=torch.DoubleTensor):
        """
        data: a Tensor.
        delta: indicator for right censoring (1 if uncensored, 0 if censored)
        lambd: regularization parameter
        """
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.seed = seed
        torch.manual_seed(seed)

        self.TType = TType

        self.n, self.p = data.shape
        n, p = self.n, self.p
        
        self.data = data.type(TType)
        self.delta = delta.type(TType)
        self.delta_dist = distmat.dist_data(self.delta, TType=TType)
        self.prev_obj = -inf

        self.beta = distmat.dist_data(torch.zeros((p,1)).type(TType), TType=TType)
        self.beta_prev = distmat.dist_data(torch.zeros((p,1)).type(TType), TType=TType)

        self.datat = self.data.t()
        
        if sigma =='power':
            l2 = self.power()
            self.sigma = (1/(2*(l2**2))).item()
        elif sigma == 'quicknorm':
            b = self.spectral_upper_bdd()
            self.sigma = (1/(2*(b**2))).item()
        else:
            self.sigma = sigma

        if self.rank == 0:
            print("step size: ", self.sigma)
        self.lambd = lambd
        self.soft_threshold = torch.nn.Softshrink(lambd)
        r_local = torch.arange(0, n).view(-1, 1).type(TType)
        r_dist = distmat.dist_data(r_local, TType=self.TType)
        self.pi_ind = ((- r_dist) + r_local.t()>= 0).type(TType)
        print(self.pi_ind.chunk)
        #print(self.rank, self.pi_ind.chunk[10,8:13])

    def l1(self):
        absdata = distmat.abs(self.data)
        return absdata.sum(dim=0).max()
    def linf(self):
        absdata = distmat.abs(self.data)
        return absdata.sum(dim=1).max()
    def spectral_upper_bdd(self):
        absdata = distmat.abs(self.data)
        r = (absdata.sum(dim=0).max()*absdata.sum(dim=1).max()).sqrt()
        return r
        
   

    

    def power(self, maxiter=1000, eps=1e-6):
        rank = self.rank
        size = self.size

        s_prev = -inf
        # match the vectors in all devices
        v = torch.rand(self.data.shape[1], 1).type(self.TType)
        X = self.data
        Xt = self.data.t()
        Xv = distmat.mm(X, v)
        if rank==0:
            print('computing max singular value...')
        for i in range(maxiter):
            v = distmat.mm(Xt, Xv)
            v/= (v**2).sum().sqrt()
            Xv = distmat.mm(X, v)
            s = (Xv**2).sum().sqrt()
            if torch.abs((s_prev - s)/s) < eps:
                break
            s_prev = s
            if i%100==0:
                if rank==0:
                    print('iteration {}'.format(i))
        if rank==0:
            print('done computing max singular value: ', s)
        return s



    def update(self):
        Xbeta = distmat.mm(self.data, self.beta)
        w = Xbeta.exp()
        W = w.cumsum(0)
        dist.barrier()

        w_dist = distmat.dist_data(w, TType=self.TType)

        pi = (w_dist/W.t()) * self.pi_ind     
        pd  = distmat.mm(pi, self.delta)
        dmpd = self.delta_dist - pd
        grad = distmat.mm(self.datat, dmpd)
        self.beta = (self.beta + grad * self.sigma).apply(self.soft_threshold)

    def get_objective(self):
        expXbeta = (distmat.mm(self.data, self.beta)).exp()
        return distmat.mm(self.delta.t(), (distmat.mm(self.data, self.beta) - (expXbeta.cumsum(0)).log())) - self.lambd * self.beta.abs().sum()

    def check_convergence(self,tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm = (self.beta_prev - self.beta).abs().max()
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol 
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm < tol
        return (diff_norm, reldiff, obj), converged

    def run(self, maxiter=100, tol=0, check_interval=1, verbose=True, check_obj=False):
        if verbose:
            if self.rank==0:
                print("Starting...")
                print("n={}, p={}".format(self.n, self.p))
                if not check_obj:
                    print("%6s\t%15s\t%10s" % ("iter", "maxdiff", "time"))
                else:
                    print("%6s\t%15s\t%15s\t%15s\t%10s" % ("iter", "maxdiff", "reldiff", "obj", "time" ))
                print('-'*80)

        t0 = time.time()
        t_start = t0

        for i in range(maxiter):
            self.beta_prev.copy_(self.beta)
            self.update()
            if (i+1) % check_interval ==0:
                t1 = time.time()
                (maxdiff, reldiff, obj), converged = self.check_convergence(tol, verbose, check_obj, check_interval=check_interval)
                if verbose:
                    if not check_obj:
                        if self.rank==0:
                            print("%6d\t%15.9e\t%10.5f" % (i+1, maxdiff, t1-t0))
                    else:
                        if self.rank==0:
                            print("%6d\t%15.9e\t%15.9e\t%15.9e\t%10.5f" % (i+1, maxdiff, reldiff,
                                                                            obj, t1-t0))
                if converged: break
                t0 = t1

        if verbose and self.rank == 0:
            print('-'*80) 
            print("Completed. total time: {}".format(time.time()-t_start))
        return self.beta

