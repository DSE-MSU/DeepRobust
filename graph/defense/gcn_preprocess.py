import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from DeepRobust.graph import utils
from DeepRobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np


class GCNSVD(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True, device='cpu'):
        super(GCNSVD, self).__init__(nfeat, nhid, nclass, dropout, with_relu, with_bias)
        self.device = device

    def fit_(self, features, adj, labels, idx_train, k=50, train_iters=200):
        modified_adj = self.truncatedSVD(adj, k=k)
        # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        self.fit(features, modified_adj, labels, idx_train, train_iters=train_iters)

    def truncatedSVD(self, data, k=50):
        print(f'=== GCN-SVD: rank={k} ===')
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print(f"rank_after = {len(S.nonzero()[0])}")
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print(f"rank_before = {len(S.nonzero()[0])}")
            diag_S = np.diag(S)
            print(f"rank_after = {len(diag_S.nonzero()[0])}")

        return U @ diag_S @ V

    def test(self, idx_test):
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

class GCNJaccard(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True, device='cpu'):
        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, with_relu, with_bias)
        self.device = device

    def fit_(self, features, adj, labels, idx_train, threshold=0.01, train_iters=200):
        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj)
        # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        self.fit(features, modified_adj, labels, idx_train, train_iters=train_iters)

    def test(self, idx_test):
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def drop_dissimilar_edges(self, features, adj):
        modified_adj = adj.copy().tolil()
        # preprocessing based on features

        print('=== GCN-Jaccrad ===')
        isSparse = sp.issparse(features)
        edges = np.array(modified_adj.nonzero()).T
        removed_cnt = 0
        for edge in tqdm(edges):
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if isSparse:
                J = self._jaccrad_similarity(features[n1], features[n2])
                if J < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                # For not binary feature, use cosine similarity
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
        print(f'removed {removed_cnt} edges in the original graph')
        return modified_adj

    def _jaccrad_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (features[n1] * features[n2]).sum()
        C = inner_product / np.sqrt(np.square(a).sum() + np.square(b).sum())
        return C
