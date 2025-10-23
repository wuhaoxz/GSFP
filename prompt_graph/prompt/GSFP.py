import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import numpy as np

class GSFP(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GSFP, self).__init__()
        self.dim = in_channels
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb
    

class GSmFP(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GSmFP, self).__init__()
        self.dim = in_channels
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()


    def add(self, x: torch.Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    
