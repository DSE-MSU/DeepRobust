import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from DeepRobust.graph import utils
from DeepRobust.graph.defense import GCN


class GCNJaccard(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, with_relu, with_bias)

    def preprocess(self):
        pass

class GCNSVD(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, with_relu, with_bias)

    def preprocess(self):
        pass

