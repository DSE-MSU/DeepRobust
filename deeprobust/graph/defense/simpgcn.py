import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from deeprobust.graph.defense import GraphConvolution
import deeprobust.graph.utils as utils
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
from itertools import product


class SimPGCN(nn.Module):
    """SimP-GCN: Node similarity preserving graph convolutional networks.
       https://arxiv.org/abs/2011.09643

    Parameters
    ----------
    nnodes : int
        number of nodes in the input grpah
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    lambda_ : float
        coefficients for SSL loss in SimP-GCN
    gamma : float
        coefficients for adaptive learnable self-loops
    bias_init : float
        bias init for the score
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train SimPGCN.
    See the detailed hyper-parameter setting in https://github.com/ChandlerBang/SimP-GCN.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import SimPGCN
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1],
        nhid=16, nclass=labels.max()+1, device='cuda')
    >>> model = model.to('cuda')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    >>> model.test(idx_test)
    """

    def __init__(self, nnodes, nfeat, nhid, nclass, dropout=0.5, lr=0.01,
            weight_decay=5e-4, lambda_=5, gamma=0.1, bias_init=0,
            with_bias=True, device=None):
        super(SimPGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.bias_init = bias_init
        self.gamma = gamma
        self.lambda_ = lambda_
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        # self.reset_parameters()
        self.scores = nn.ParameterList()
        self.scores.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(1):
            self.scores.append(Parameter(torch.FloatTensor(nhid, 1)))

        self.bias = nn.ParameterList()
        self.bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.bias.append(Parameter(torch.FloatTensor(1)))

        self.D_k = nn.ParameterList()
        self.D_k.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(1):
            self.D_k.append(Parameter(torch.FloatTensor(nhid, 1)))

        self.identity = utils.sparse_mx_to_torch_sparse_tensor(
                sp.eye(nnodes)).to(device)

        self.D_bias = nn.ParameterList()
        self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.D_bias.append(Parameter(torch.FloatTensor(1)))

        # discriminator for ssl
        self.linear = nn.Linear(nhid, 1).to(device)

        self.adj_knn = None
        self.pseudo_labels = None

    def get_knn_graph(self, features, k=20):
        if not os.path.exists('saved_knn/'):
           os.mkdir('saved_knn')
        if not os.path.exists('saved_knn/knn_graph_{}.npz'.format(features.shape)):
            features[features!=0] = 1
            sims = cosine_similarity(features)
            np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)

            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            adj_knn = sp.csr_matrix(sims)
            sp.save_npz('saved_knn/knn_graph_{}.npz'.format(features.shape), adj_knn)
        else:
            print('loading saved_knn/knn_graph_{}.npz...'.format(features.shape))
            adj_knn = sp.load_npz('saved_knn/knn_graph_{}.npz'.format(features.shape))
        return preprocess_adj_noloop(adj_knn, self.device)

    def initialize(self):
        """Initialize parameters of SimPGCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

        for s in self.scores:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.bias:
            # fill in b with postive value to make
            # score s closer to 1 at the beginning
            b.data.fill_(self.bias_init)

        for Dk in self.D_k:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)

        for b in self.D_bias:
            b.data.fill_(0)


    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)


    def forward(self, fea, adj):
        x, _ = self.myforward(fea, adj)
        return x

    def myforward(self, fea, adj):
        '''output embedding and log_softmax'''
        if self.adj_knn is None:
            self.adj_knn = self.get_knn_graph(fea.to_dense().cpu().numpy())

        adj_knn = self.adj_knn
        gamma = self.gamma

        s_i = torch.sigmoid(fea @ self.scores[0] + self.bias[0])

        Dk_i = (fea @ self.D_k[0] + self.D_bias[0])
        x = (s_i * self.gc1(fea, adj) + (1-s_i) * self.gc1(fea, adj_knn)) + (gamma) * Dk_i * self.gc1(fea, self.identity)

        x = F.dropout(x, self.dropout, training=self.training)
        embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
        x = (s_o * self.gc2(x, adj) + (1-s_o) * self.gc2(x, adj_knn)) + (gamma) * Dk_o * self.gc2(x, self.identity)

        x = F.log_softmax(x, dim=1)

        self.ss = torch.cat((s_i.view(1,-1), s_o.view(1,-1), gamma*Dk_i.view(1,-1), gamma*Dk_o.view(1,-1)), dim=0)
        return x, embedding

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.features.to_dense())
            self.pseudo_labels = agent.get_label().to(self.device)
            node_pairs = agent.node_pairs
            self.node_pairs = node_pairs

        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        # print(loss)
        return loss

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output


    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):

            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class AttrSim:

    def __init__(self, features):
        self.features = features.cpu().numpy()
        self.features[self.features!=0] = 1

    def get_label(self, k=5):
        features = self.features
        if not os.path.exists('saved_knn/cosine_sims_{}.npy'.format(features.shape)):
            sims = cosine_similarity(features)
            np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)
        else:
            print('loading saved_knn/cosine_sims_{}.npy'.format(features.shape))
            sims = np.load('saved_knn/cosine_sims_{}.npy'.format(features.shape))

        if not os.path.exists('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape)):
            try:
                indices_sorted = sims.argsort(1)
                idx = np.arange(k, sims.shape[0]-k)
                selected = np.hstack((indices_sorted[:, :k],
                    indices_sorted[:, -k-1:]))

                selected_set = set()
                for i in range(len(sims)):
                    for pair in product([i], selected[i]):
                        if pair[0] > pair[1]:
                            pair = (pair[1], pair[0])
                        if  pair[0] == pair[1]:
                            continue
                        selected_set.add(pair)

            except MemoryError:
                selected_set = set()
                for ii, row in tqdm(enumerate(sims)):
                    row = row.argsort()
                    idx = np.arange(k, sims.shape[0]-k)
                    sampled = np.random.choice(idx, k, replace=False)
                    for node in np.hstack((row[:k], row[-k-1:], row[sampled])):
                        if ii > node:
                            pair = (node, ii)
                        else:
                            pair = (ii, node)
                        selected_set.add(pair)

            sampled = np.array(list(selected_set)).transpose()
            np.save('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape), sampled)
        else:
            print('loading saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape))
            sampled = np.load('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape))
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])
        self.sims = sims
        return torch.FloatTensor(sims[self.node_pairs]).reshape(-1,1)


def preprocess_adj_noloop(adj, device):
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = utils.sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

