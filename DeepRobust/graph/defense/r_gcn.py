'''
    Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.
        http://pengcui.thumedialab.com/papers/RGCN.pdf
    Author's Tensorflow implemention:
        https://github.com/thumanlab/nrlweb/tree/master/static/assets/download
'''

import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.distributions.multivariate_normal import MultivariateNormal
from DeepRobust.graph import utils
import torch.optim as optim
from copy import deepcopy

class GaussianConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GaussianConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.sigma = Parameter(torch.FloatTensor(out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, previous_miu, previous_sigma, adj_norm1=None, adj_norm2=None, gamma=1):

        if adj_norm1 is None and adj_norm2 is None:
            return torch.mm(previous_miu, self.weight_miu), \
                    torch.mm(previous_sigma, self.weight_sigma)

        Att = torch.exp(-gamma * previous_sigma)
        M = adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

        # M = torch.mm(torch.mm(adj, previous_miu * A), self.weight_miu)
        # Sigma = torch.mm(torch.mm(adj, previous_sigma * A * A), self.weight_sigma)

        # TODO sparse implemention
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        # return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RGCN(Module):

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=5e-4, beta2=5e-4, lr=0.01, dropout=0.6, device='cpu'):
        super(RGCN, self).__init__()

        self.device = device
        # adj_norm = normalize(adj)
        # first turn original features to distribution
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid
        self.gc1 = GaussianConvolution(nfeat, nhid)
        # self.gc2 = GaussianConvolution(nhid, nhid)
        # self.gc3 = GaussianConvolution(nhid, nclass)
        self.gc2 = GaussianConvolution(nhid, nclass)

        self.dropout = dropout
        # self.gaussian = MultivariateNormal(torch.zeros(self.nclass), torch.eye(self.nclass))
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, self.nclass),
                torch.diag_embed(torch.ones(nnodes, self.nclass)))
        self.miu1 = None
        self.sigma1 = None
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self):
        features = self.features
        miu, sigma = self.gc1(features, features)
        miu, sigma = F.elu(miu, alpha=1), F.relu(sigma)
        self.miu1, self.sigma1 = miu, sigma
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)

        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        miu, sigma = F.elu(miu, alpha=1), F.relu(sigma)

        # # third layer
        # miu = F.dropout(miu, self.dropout, training=self.training)
        # sigma = F.dropout(sigma, self.dropout, training=self.training)
        # miu, sigma = self.gc3(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        # miu, sigma = F.elu(miu), F.relu(sigma)
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)
        return F.log_softmax(output, dim=1)

    def fit_(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=True):

        adj, features, labels = utils.to_tensor(adj.todense(), features, labels, device=self.device)
        self.features, self.labels = features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1/2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        print('=== training rgcn model ===')
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self.loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self.loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output

        print('=== picking the best model according to the performance on validation ===')


    def test(self, idx_test):
        # output = self.forward()
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        kl_loss = 0.5 * (self.miu1.pow(2) + self.sigma1 - torch.log(1e-8 + self.sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
               torch.norm(self.gc1.weight_sigma, 2).pow(2)
        # print(f'gcn_loss: {loss.item()}, kl_loss: {self.beta1 * kl_loss.item()}, norm2: {self.beta2 * norm2.item()}')
        return loss  + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1/2):

        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj)).to(self.device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power

