import torch
from .pet_utils import *
from . import distmat
import torch.distributed as dist
class PET():
    def __init__(self, y, emat, G, D, TType=torch.DoubleTensor, mu=1e-6, eps=1e-20):
        """
        y: count data, d x 1 (broadcasted tensor)
        emat: "E" matrix generated by E_generate, d x (p) dense (or sparse?)
        g: neighboorhood matrix (sparse), (p) x p sparse
        d: difference matrix (sparse) to define loss, (#edges) x (p) sparse
        mu: smoothness parameter
        """
        assert not emat.byrow
        self.TType = TType
        self.d = emat.shape[0]
        self.p = emat.shape[1]
        self.y = y.type(TType).view(-1, 1)
        #self.emat = emat.type(TType)
        self.mu = mu
        self.eps = eps

        emat_typename = torch.typename(emat.chunk).split('.')[-1]
        if emat.chunk.is_cuda:
            SpTType = getattr(torch.cuda.sparse, emat_typename)
        else:
            SpTType = getattr(torch.sparse, emat_typename)

        if emat.chunk.layout == torch.sparse_coo:
            self.emat = emat.type(SpTType)
            #self.Et = self.E.t().coalesce()
            self.emat = self.emat.coalesce()
        else:
            self.emat = emat.type(TType)
            #self.Et = self.E.t()

        self.G = G.type(SpTType)
        self.D = D.type(SpTType)
        assert G.byrow
        assert not D.byrow

        # scale E to have unit l1 column norms : already done in generator
        # copute |N_j|  = G @ 1
        self.N = distmat.mm(self.G, torch.ones(G.shape[1], 1).type(TType))
        # a_j = -2 * mu * |N_j|, j = 1, ..., p
        self.a  = -2 * self.mu * self.N 
        # initialize: lambda_j = 1, j = 1, ..., p
        self.lambd = distmat.distgen_ones(G.shape[0], 1).type(TType)
        self.lambd_prev = distmat.distgen_ones(G.shape[0], 1).type(TType)

        
        self.prev_obj=inf

    def update(self):
        # update z
        el = distmat.mm(self.emat, self.lambd) 
        gl = distmat.mm(self.G, self.lambd)
        z = self.emat * self.y * self.lambd.t()/(el + self.eps)
        # update b
        b = self.mu * (self.N*self.lambd + gl) -1
        # update c
        c = z.sum(dim=0).t()
        # update lambda
        if self.mu != 0:
           
            self.lambd = (-b - (b**2 - 4* self.a * c).sqrt())/(2 * self.a + self.eps)
        else:
            self.lambd = -c/(b+self.eps)
    def get_objective(self):
        el = distmat.mm(self.emat, self.lambd)
        likelihood = (self.y * torch.log(el+self.eps) - el).sum()
        dl = distmat.mm(self.D, self.lambd)
        penalty = - self.mu/2.0 * torch.sum(dl**2)
        return  likelihood + penalty
    def check_convergence(self, tol, verbose=True, check_obj=False, check_interval=1):
        obj = None
        diff_norm = ((self.lambd_prev - self.lambd).abs()).max()
        if check_obj:
            obj = self.get_objective()
            reldiff = abs(self.prev_obj - obj)/((abs(obj)+1)*check_interval)
            converged = reldiff < tol
            self.prev_obj = obj
        else:
            reldiff = None
            converged = diff_norm < tol
        return (diff_norm, reldiff, obj), converged
    def run(self, maxiter=1000, tol=1e-5, check_interval=1, verbose=True, check_obj=False):
        rank = dist.get_rank()
        if verbose:
            if rank==0:
                print("Starting...")
                print("d={}, p={}".format(self.d, self.p))
                if not check_obj:
                    print("%6s\t%13s\t%15s" % ("iter", "maxdiff", "time"))
                else:
                    print("%6s\t%13s\t%15s\t%15s\t%10s" % ("iter", "maxdiff", "reldiff", "obj", "time" ))
                print('-'*80)
        t0 = time.time()
        t_start = t0

        for i in range(maxiter):
            self.lambd_prev.copy_(self.lambd)
            self.update()
            if (i+1) % check_interval ==0:
                t1 = time.time()
                (maxdiff, reldiff, obj), converged = self.check_convergence(tol, verbose, check_obj, 
                                                                            check_interval)
                if verbose:
                    if not check_obj:
                        if rank==0:
                            print("%6d\t%13.4e\t%10.5f" % (i+1, maxdiff, t1-t0))
                    else:
                        if rank==0:
                            print("%6d\t%13.4e\t%15.9e\t%15.9e\t%10.5f" % (i+1, maxdiff, reldiff,
                                                                            obj, t1-t0))
                if converged: break
                t0 = t1
            dist.barrier()

        if verbose:
            if rank==0:
                print('-'*80)
                print("Completed. total time: {}".format(time.time()-t_start))
