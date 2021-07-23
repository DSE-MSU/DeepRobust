"""
    Adversarial Attacks on Graph Neural Networks via Meta Learning. ICLR 2019
        https://openreview.net/pdf?id=Bylnx209YX
    Author Tensorflow implementation:
        https://github.com/danielzuegner/gnn-meta-attack
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack


class BaseMeta(BaseAttack):
    """Abstract base class for meta attack. Adversarial Attacks on Graph Neural
    Networks via Meta Learning, ICLR 2019,
    https://openreview.net/pdf?id=Bylnx209YX

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'

    """

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class Metattack(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


class MetaApprox(BaseMeta):
    """Approximated version of Meta Attack. Adversarial Attacks on
    Graph Neural Networks via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MetaApprox
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=True)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):

        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.lr = lr
        self.train_iters = train_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(nnodes, nnodes).to(device)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape).to(device)

        self.with_bias = with_bias

        self.weights = []
        self.biases = []

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr) # , weight_decay=5e-4)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            # w.data.fill_(1)
            # b.data.fill_(1)
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            # hidden = features
            # for w, b in zip(self.weights, self.biases):
            #     if self.sparse_features:
            #         hidden = adj_norm @ torch.spmm(hidden, w) + b
            #     else:
            #         hidden = adj_norm @ hidden @ w + b
            #     if self.with_relu:
            #         hidden = F.relu(hidden)

            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

            self.optimizer.step()


        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))


    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):
        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            self._initialize()

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)

            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()
