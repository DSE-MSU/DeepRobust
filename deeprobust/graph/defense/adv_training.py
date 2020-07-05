import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np


class AdvTraining:
    """Adversarial training framework for defending against attacks.

    Parameters
    ----------
    model :
        model to protect, e.g, GCN
    adversary :
        attack model
    device : str
        'cpu' or 'cuda'
    """

    def __init__(self, model, adversary=None, device='cpu'):

        self.model = model
        if adversary is None:
            adversary = RND()
        self.adversary = adversary
        self.device = device

    def adv_train(self, features, adj, labels, idx_train, train_iters, **kwargs):
        """Start adversarial training.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        """
        for i in range(train_iters):
            modified_adj = self.adversary.attack(features, adj)
            self.model.fit(features, modified_adj, train_iters, initialize=False)




