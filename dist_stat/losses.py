import numpy as np
import torch

class LossFunction:
    @classmethod
    def eval(self, bhat, b):
        pass
    @classmethod
    def eval_deriv(self, bhat, b):
        pass

class QuadLoss(LossFunction):
    @classmethod
    def eval(self, bhat, b):
        return 0.5 * torch.norm(bhat-b)**2
    @classmethod
    def eval_deriv(self, bhat, b):
        return bhat - b

class LogisticLoss(LossFunction):
    @classmethod
    def eval(self, bhat, b):
        return torch.sum(torch.log(1+torch.exp(-b*bhat)))
    @classmethod
    def eval_deriv(self, bhat, b):
        expbAx = torch.exp(b*bhat)
        return -b/(1.0 + expbAx)
