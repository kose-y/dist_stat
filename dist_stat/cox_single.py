import torch
import time
from math import inf
"""
l1-regularized cox regression
"""

class COX():
    def __init__(self, data, delta, lambd, time=None, sigma=None, TType=torch.DoubleTensor):
        """
        data: a Tensor.
        delta: indicator for right censoring (1 if uncensored, 0 if censored)
        lambd: regularization parameter
        """
        self.TType = TType

        self.n, self.p = data.shape
        n, p = self.n, self.p
        
        self.data = data.type(TType)
        self.delta = delta.type(TType)
        self.prev_obj = -inf

        self.beta = torch.zeros((p,1)).type(TType)
        self.beta_prev = torch.zeros((p,1)).type(TType)

        self.datat = self.data.t()
        
        if sigma is None:
            self.sigma = 1/(2*self.power()**2)
        else:
            self.sigma = sigma
        self.lambd = lambd
        self.soft_threshold = torch.nn.Softshrink(lambd)
        if time is None:
            time = -torch.arange(0, n).view(-1, 1)
        self.pi_ind = (time.t() - time >= 0).type(TType)
            

    def power(self, maxiter=1000, eps=1e-6):
        s_prev = -inf
        v = torch.rand(self.data.shape[1], 1).type(self.TType)
        X = self.data
        Xt = self.data.t()
        Xv = torch.mm(X, v)
        print('computing max singular value...')
        for i in range(maxiter):
            v = torch.mm(Xt, Xv)
            v/= v.norm()
            Xv = torch.mm(X, v)
            s = torch.norm(Xv)
            if torch.abs((s_prev - s)/s) < eps:
                break
            s_prev = s
            if i%100==0:
                print('iteration {}'.format(i))
        print('done computing max singular value')
        return s



    def update(self):
        Xbeta = torch.mm(self.data, self.beta)
        w = torch.exp(Xbeta)
        #W = torch.mm(self.pi_ind, w)
      
        W = torch.cumsum(w, 0)

        #zz = (torch.cumsum(w * self.data, 0)/W).t()
   
        #grad = torch.mm(self.datat - zz , self.delta)

        pi = self.pi_ind.t() * w / W.t()
        grad = torch.mm(self.datat, self.delta - torch.mm(pi, self.delta))
        self.beta = self.soft_threshold(self.beta + self.sigma * grad)
        

    def get_objective(self):
        expXbeta = torch.exp(torch.mm(self.data, self.beta))
        return torch.mm(self.delta.t(), (torch.mm(self.data, self.beta) - torch.log(torch.mm(self.pi_ind, expXbeta)))) - self.lambd * torch.sum(torch.abs(self.beta))

    def check_convergence(self,tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm = torch.max(torch.abs(self.beta_prev - self.beta))
        nonzeros = (self.beta != 0).sum()
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol 
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm < tol
        return (diff_norm, reldiff, obj, nonzeros), converged

    def run(self, maxiter=100, tol=1e-5, check_interval=1, verbose=True, check_obj=False):
        if verbose:
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
                        print("%6d\t%15.9e\t%15d\t%10.5f" % (i+1, maxdiff, nonzeros, t1-t0))
                    else:
                        print("%6d\t%15.9e\t%15.9e\t%15.9e\t%15d\t%10.5f" % (i+1, maxdiff, reldiff,
                                                                            obj, nonzeros, t1-t0))
                if converged: break
                t0 = t1

        if verbose:
            print('-'*80) 
            print("Completed. total time: {}".format(time.time()-t_start))

