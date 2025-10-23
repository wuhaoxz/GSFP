from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from torch.optim import Optimizer
import torch
import sklearn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pdb

class PGD(Optimizer):

    def __init__(self, params, proxs, lr=required, momentum=0, dampening=0, weight_decay=0, alphas=[]):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)


        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']


            for param in group['params']:              
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha)
                    

                    


class ProxOperators():

    def __init__(self):
        self.nuclear_norm = None

    def prox_l21(self, data, alpha):
        data =data.T.contiguous() 
        q_norm = torch.norm(data, dim=1)
        q_norm1 = (q_norm-alpha)/q_norm
        s_value = torch.where(alpha < q_norm, q_norm1, torch.zeros_like(q_norm1))
        data = torch.mm(torch.diag(s_value), data)
        data =data.T.contiguous() 
        return data

    def prox_l1(self, data, alpha):
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data



prox_operators = ProxOperators()

