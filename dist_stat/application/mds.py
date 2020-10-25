import torch
import torch.distributed as dist
from .. import distmat
from ..distmat import THDistMat, distgen_uniform, distgen_base, linf_diff, l2_diff
import time
from .euclidean_distance import euclidean_distance_DistMat
from math import inf
"""
MDS as in Zhou et al.
"""
class MDS():
    def __init__(self, data, p, is_distance=False, weights=1, TType=torch.DoubleTensor, init_from_master=False, verbose=False):
        """
        data: a THDistMat, byrow, q x q distance matrix 
        p: a small integer.
        weights: a scalar or a symmetric q x q tensor (THDistMat)
        is_distance: if the data is already a computed distance matrix
        """
        assert data.byrow
        self.TType = TType
        if is_distance:
            assert data.shape[0] == data.shape[1]
            self.y = data
        else:
            if verbose:
                print("computing all pairwise distance...")
                t0 =  time.time()
            self.y = euclidean_distance_DistMat(data).type(TType)
            if verbose:
                t1 = time.time()
                print("took {} seconds.".format(t1-t0))

        self.p = p 
        self.q = data.shape[0]
 

        self.weights = weights

        if isinstance(weights, THDistMat):
            self.w_sums = self.weights.sum(1) 
        elif torch.is_tensor(weights):
            raise NotImplementedError("an atomic value or a THDistMat is accepted for weights")
        else:
            self.weights = float(weights)
            self.w_sums = self.weights * float(self.q - 1)
        self.x = distmat.mul(self.y, weights).type(TType)
        
        self.theta      = distgen_uniform(self.q, self.p, lo=-1, hi=1, set_from_master=init_from_master, TType=TType)
        self.theta_prev = distgen_uniform(self.q, self.p, lo=-1, hi=1, set_from_master=init_from_master, TType=TType)
        self.prev_obj = inf


    def update(self):
        """
        one iteration of MDS.
        """
        d   = distmat.mm(self.theta, self.theta.t())
        TtT_diag_dist = d.diag(distribute=False).view(1,-1) # to broadcast it
        TtT_diag      = d.diag(distribute=True)
        d   = d.mul_(-2.0)
        d.add_(TtT_diag_dist)  
        d.add_(TtT_diag)

        d.fill_diag_(inf)
        Z      = distmat.div(self.x, d)
        Z_sums = Z.sum(dim=1) # length-q vector
        WmZ = distmat.sub(self.weights, Z)

        if isinstance(self.weights, float):
            WmZ.fill_diag_(0)

        TWmZ = distmat.mm(self.theta.t(), WmZ.t())# this is okay, since W-Z is symmetric, out_sizes=self.theta.sizes)
        self.theta = (self.theta * (self.w_sums + Z_sums) + TWmZ.t())/(self.w_sums * 2.0)
    def get_objective(self):
        """ 
        returns the objective function
        """
        distances = euclidean_distance_DistMat(self.theta)
        return ((self.y - distances)**2 * self.weights).sum()/2.0
    def check_convergence(self, tol, verbose=True, check_obj=False, check_interval=1):
        rank = dist.get_rank()
        obj = None
        diff_norm = linf_diff(self.theta_prev, self.theta)
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm < tol
        return (diff_norm, reldiff, obj), converged
        
    def run(self, maxiter=100, tol=1e-5, check_interval=1, verbose=True, check_obj=False):
        rank = dist.get_rank()
        if verbose:
            if rank==0:
                print("Starting...")
                print("p={}, q={}".format(self.p, self.q))
                if not check_obj:
                    print("%6s\t%13s\t%10s" % ("iter", "theta_maxdiff", "time"))
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
                        if rank ==0:
                            print("%6d\t%13.4e\t%10.5f" % (i+1, theta_maxdiff, t1-t0))
                    else:
                        if rank==0:
                            print("%6d\t%13.4e\t%15.9e\t%15.9e\t%10.5f" % (i+1, theta_maxdiff,  
                                                                            reldiff, obj, t1-t0))
                if converged: break
                t0 = t1

            dist.barrier()
        if verbose:
            if rank==0:
                print('-'*80) 
                print("Completed. total time: {}".format(time.time()-t_start))
        return self.theta


