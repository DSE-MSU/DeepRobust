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
    '''
        Adversarial training Framework for defending against attacks
    '''
    def __init__(self, model, adversary=None, device='cpu'):

        self.model = model
        if adversary is None:
            adversary = RND()
        self.adversary = adversary
        self.device = device

    def adv_train(self, features, adj, labels, idx_train, train_iter):
        for i in range(train_iter):
            modified_adj = self.adversary.attack(features, adj)
            self.model.fit(features, modified_adj, train_iter, initialize=False)




