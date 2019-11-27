import torch
import torch.distributed as dist
import time
from .euclidean_distance import euclidean_distance_tensor
from math import inf
"""
MDS as in Zhou et al.
"""
class MDS():
    def __init__(self, data, p, is_distance=False, weights=1, TType=torch.DoubleTensor, verbose=False):
        """
        data: a tensor q x q distance matrix 
        p: a small integer.
        weights: a scalar or a q x q tensor (THDistMat)
        is_distance: if the data is already a computed distance matrix
        """
        self.TType = TType
        if is_distance:
            assert data.shape[0] == data.shape[1]
            self.y = data.type(TType)
        else:
            if verbose:
                print("computing all pairwise distance...")
                t0 =  time.time()
            dt = data.type(TType)
            self.y = euclidean_distance_tensor(dt, dt)
            if verbose:
                t1 = time.time()
                print("took {} seconds.".format(t1-t0))

        self.p = p 
        self.q = data.shape[0]

        self.weights = weights
        self.x = torch.mul(self.y, self.weights)

        if torch.is_tensor(weights):
            self.w_sums = self.weights.sum(dim=1, keepdim=True)
        else:
            self.weights = float(weights)
            self.w_sums = self.weights*float(self.q-1) # zero on diagonals

        self.theta = torch.DoubleTensor(self.q, p).uniform_(-1, 1).type(TType)
        #firsts = self.theta.view(-1)[:(2*self.q-1)] # first 2 * self.q -1 elements to zero to avoid ambiguitiy
        #firsts.fill_(0)
        
        self.theta_prev = TType(self.q, p)
        self.prev_obj = inf

    def update(self):
        """
        one iteration of MDS.
        """
        d   = torch.mm(self.theta, self.theta.t())
        TtT_diag = torch.diag(d).view(-1, 1) # to broadcast it
        d   = d.mul_(-2.0)
        d.add_(TtT_diag)  
        d.add_(TtT_diag.t())

        # directly modify the diagonal
        d_diag = d.view(-1)[::(self.q+1)]
        d_diag.fill_(inf)
        Z      = torch.div(self.x, d)
        
        Z_sums = Z.sum(dim=1, keepdim=True) # length-q vector

        WmZ = self.weights-Z
        if isinstance(self.weights, float):
            WmZ_diag = WmZ.view(-1)[::(self.q+1)]
            WmZ_diag.fill_(0)

        TWmZ = torch.mm(self.theta.t(), WmZ)

        self.theta = (self.theta * (self.w_sums + Z_sums) + TWmZ.t())/(self.w_sums * 2.0)

    def get_objective(self):
        """ 
        returns the objective function
        """
        distances = euclidean_distance_tensor(self.theta, self.theta)
        return (((self.y - distances)*self.weights)**2).sum()/2.0 
    def check_convergence(self, tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm = torch.max(self.theta_prev-self.theta)
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm <tol

        return (diff_norm, reldiff, obj), converged
        
    def run(self, maxiter=100, tol=1e-5, check_interval=1, verbose=True, check_obj=False):
        if verbose:
            print("Starting...")
            print("p={}, q={}".format(self.p, self.q))
            if not check_obj:
                print("%6s\t%13s\t%15s" % ("iter", "theta_maxdiff", "time"))
            else:
                print("%6s\t%13s\t%15s\t%15s\t%10s" % ("iter", "theta_maxdiff", "reldiff", "obj", "time" ))
            print('-'*80)

        t0 = time.time()
        t_start = t0

        for i in range(maxiter):
            self.theta_prev.copy_(self.theta)
            
            self.update()
            if (i+1) % check_interval ==0:
                t1 = time.time()
                (theta_maxdiff, reldiff, obj), converged = self.check_convergence(tol, verbose, check_obj, check_interval)
                if verbose:
                    if not check_obj:
                        print("%6d\t%13.4e\t%10.5f" % (i+1, theta_maxdiff, t1-t0))
                    else:
                        print("%6d\t%13.4e\t%15.9e\t%15.9e\t%10.5f" % (i+1, theta_maxdiff, reldiff, 
                                                                            obj, t1-t0))
                if converged: break
                t0 = t1

        if verbose:
            print('-'*80) 
            print("Completed. total time: {}".format(time.time()-t_start))


