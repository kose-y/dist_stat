import torch
import torch.distributed as dist
import time
from math import inf
from . import distmat
import os
from .utils import breslow_ind
"""
l1-regularized cox regression
"""

class COX():
    def __init__(self, data, delta, lambd, time=None, seed=16962, sigma='power', TType=torch.DoubleTensor):
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


        print(self.sigma)
        self.lambd = lambd
        self.soft_threshold = torch.nn.Softshrink(lambd)

        if time is None:
            time = -torch.arange(0,n).view(-1,1)

        self.breslow_ind = torch.tensor(breslow_ind(time.cpu().numpy())).to(dtype=torch.int64, device=self.beta.chunk.device)

        time_local = time.reshape(-1, 1).type(TType)
        time_dist = distmat.dist_data(time, TType=self.TType)
        #r_local = torch.arange(0, n).view(-1, 1).type(TType)
        #r_dist = distmat.dist_data(r_local, TType=self.TType)
        self.pi_ind = (time_dist - time_local.t()>= 0).type(TType)
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
        print(r)
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
            print('done computing max singular value')
        print(s)
        return s



    def update(self):
        Xbeta = distmat.mm(self.data, self.beta)
        w = Xbeta.exp()
        ###print("%20.9e" % (w.max()), file=self.devnull)
        #print("%20.9e" % (w.sum()))
        W = w.cumsum(0)[self.breslow_ind]
        dist.barrier()
        #print("%20.9e" % (W.max()))
        #print(W, file=self.devnull)
        # pi = (w / W.t()) * self.pi_ind

        w_dist = distmat.dist_data(w, TType=self.TType)
        ###print("%20.9e" % (w_dist.max()), file=self.devnull)
        #print("%20.9e" % (w_dist.sum()))

        pi = self.pi_ind * (w_dist/W.t())    
        ###print(pi.sum(), file=self.devnull)
        #print(pi.max())
        #print(pi.min())

        #print((w/W.t()).chunk[:10, :10], self.rank)
        #if self.size==1:
        #    print((w/W.t()).chunk[5000:5010, 5000:5010], 1) 
        #print(pi.chunk[:10, :10], self.rank)
        #if self.size==1:
        #    print(pi.chunk[5000:5010, 5000:5010], 1)
        #print((pi**2).sum()) 
        pd  = distmat.mm(pi, self.delta)
        ###print("%20.9e" % (pd.max()), file=self.devnull)
        #print("%20.9e" % (pd.sum()))
        #print("%20.9e" % ((pd**2).sum()))
        dmpd = self.delta_dist - pd
        ###print("%20.9e" % (dmpd.max()), file=self.devnull)
        #print("%20.9e" % (dmpd.sum()))
        #print("%20.9e" % ((dmpd**2).max()))
        grad = distmat.mm(self.datat, dmpd)
        #print(grad.max())
        ###print("%20.9e" % (grad.max()), file=self.devnull)
        #print("%20.9e" % (grad.sum()))
        #print("%20.9e" % ((grad**2).sum()))
        

        #self.beta = self.soft_threshold(self.beta + grad * self.sigma)
        self.beta = (self.beta + grad * self.sigma).apply(self.soft_threshold)
        #print(self.beta.max())       

    def get_objective(self):
        expXbeta = (distmat.mm(self.data, self.beta)).exp()
        return distmat.mm(self.delta.t(), (distmat.mm(self.data, self.beta) - (expXbeta.cumsum(0)[self.breslow_ind]).log())) - self.lambd * self.beta.abs().sum()

    def check_convergence(self,tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm = (self.beta_prev - self.beta).abs().max()
        nonzeros = (self.beta != 0).type(torch.int64).sum()
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol 
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm < tol
        return (diff_norm, reldiff, obj, nonzeros), converged

    def run(self, maxiter=100, tol=0, check_interval=1, verbose=True, check_obj=False):
        if verbose:
            if self.rank==0:
                print("Starting...")
                print("n={}, p={}".format(self.n, self.p))
                if not check_obj:
                    print("%6s\t%15s\t%15s\t%10s" % ("iter", "maxdiff", "nonzeros", "time"))
                else:
                    print("%6s\t%15s\t%15s\t%15s\t%15s\t%10s" % ("iter", "maxdiff", "reldiff", "obj", "nonzeros", "time" ))
                print('-'*80)

        t0 = time.time()
        t_start = t0

        for i in range(maxiter):
            self.beta_prev.copy_(self.beta)
            self.update()
            if (i+1) % check_interval ==0:
                t1 = time.time()
                (maxdiff, reldiff, obj, nonzeros), converged = self.check_convergence(tol, verbose, check_obj, check_interval=check_interval)
                if verbose:
                    if not check_obj:
                        if self.rank==0:
                            print("%6d\t%15.9e\t%15d\t%10.5f" % (i+1, maxdiff, nonzeros, t1-t0))
                    else:
                        if self.rank==0:
                            print("%6d\t%15.9e\t%15.9e\t%15.9e\t%15d\t%10.5f" % (i+1, maxdiff, reldiff,
                                                                            obj, nonzeros, t1-t0))
                if converged: break
                t0 = t1

        if verbose:
            print('-'*80) 
            print("Completed. total time: {}".format(time.time()-t_start))

