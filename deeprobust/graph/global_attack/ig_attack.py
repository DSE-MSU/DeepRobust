'''
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
'''

import torch
from deeprobust.graph.global_attack import BaseAttack
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp

class IGAttack(BaseAttack):

    def __init__(self, model=None, nnodes=None, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(IGAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, ori_features, ori_adj, labels, idx_train, perturbations):
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        modified_adj = ori_adj

        victim_model.eval()

        s_e = self.calc_importance_edge(ori_features, ori_adj, labels, idx_train, steps=10)

        import ipdb
        ipdb.set_trace()

        for t in tqdm(range(perturbations)):
            modified_adj

        self.adj_changes.data.copy_(torch.tensor(best_s))
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

    def calc_importance_edge(self, features, adj, labels, idx_train, steps):
        adj.requires_grad = True
        integrated_grad_list = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j]:
                    scaled_inputs = [(float(i)/ steps) * (adj - 0) for i in range(0, steps+1)]
                else:
                    scaled_inputs = [-(float(i)/ steps) * (1 - adj) for i in range(0, steps + 1)]
                _sum = 0
                for new_adj in scaled_inputs:
                    # TODO: whether to first normalize adj or
                    adj_norm = utils.normalize_adj_tensor(new_adj)
                    output = self.surrogate(features, adj_norm)
                    loss = F.nll_loss(output[idx_train], labels[idx_train])

                    import ipdb
                    ipdb.set_trace()

                    # adj_grad = torch.autograd.grad(loss, adj[i][j], allow_unused=True)[0]
                    adj_grad = torch.autograd.grad(loss, adj, allow_unused=True)[0]
                    adj_grad = adj_grad[i][j]
                    _sum += adj_grad

                avg_grads = _sum.mean()
                integrated_grad_list.append(avg_grad)

        return integrated_grad_list

    def calc_importance_feature(self, input, steps):
        integrated_grad_list = []
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if input[i][j]:
                    scaled_inputs = [(float(k)/ steps) * (input - 0) for k in ragen(0, steps+1)]
                else:
                    scaled_inputs = [-(float(k)/ steps) * (1 - input) for k in ragen(0, steps + 1)]

                avg_grads = self.calc_gradient_feature(scaled_inputs, model, target_label_idx, cuda)
                integrated_grad_list.append(avg_grad)

        return integrated_grad_list

    def calc_gradient_adj(self, inputs, feature):
        for adj in inputs:
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = self.surrogate(features, adj_norm)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, inputs)[0]
        return adj_grad.mean()

    def calc_gradient_feature(self, adj_norm, inputs):
        for features in inputs:
            output = self.surrogate(features, adj_norm)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, inputs)[0]
        return adj_grad.mean()

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        ind = np.diag_indices(self.adj_changes.shape[0])
        adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
        modified_adj = adj_changes_symm + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

