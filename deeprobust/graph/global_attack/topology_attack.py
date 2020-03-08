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

class PGDAttack(BaseAttack):

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, perturbations):
        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        victim_model.eval()
        epochs = 200
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

    def random_sample(self, ori_adj, ori_features, labels, idx_train, perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                print(sampled.sum())
                if sampled.sum() > perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = F.nll_loss(output[idx_train], labels[idx_train])
                print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):

        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes-1, col=self.nnodes-1, offset=0)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        # m += m.t()
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def bisection(self, a, b, perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


class MinMax(PGDAttack):

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features, device=device)


    def attack(self, ori_features, ori_adj, labels, idx_train, perturbations):
        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        # optimizer
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        epochs = 200
        victim_model.eval()
        for t in tqdm(range(epochs)):
            # update victim model
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate pgd attack
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            # adj_grad = self.adj_changes.grad

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            # self.adj_changes.grad.zero_()
            self.projection(perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

