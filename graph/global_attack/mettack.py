'''
    Adversarial Attacks on Graph Neural Networks via Meta Learning. ICLR 2019
        https://openreview.net/pdf?id=Bylnx209YX
    Author Tensorflow implementation:
        https://github.com/danielzuegner/gnn-meta-attack
'''

import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from DeepRobust.graph import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp

from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp

class BaseMeta(BaseAttack):

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_
        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

        if attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.features.data.fill_(0)
        else:
            pass

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
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
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)
        return allowed_mask, current_ratio


class Metattack(BaseMeta):

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_features, lambda_, device)
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
                if self.with_relu:
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
            if self.with_relu:
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

        print(f'GCN loss on unlabled data: {loss_test_val.item()}')
        print(f'GCN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')
        print(f'attack loss: {attack_loss.item()}')
        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        # adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, create_graph=True)[0]
        return adj_grad


    def attack(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(features)

        labels_self_training = self.self_training_label(labels, idx_train)

        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad = self.get_meta_grad(features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_grad = adj_grad * (-2 * modified_adj + 1)

            # Make sure that the minimum entry is 0.
            adj_meta_grad -= adj_meta_grad.min()

            # Filter self-loops
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))

            # # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            ll_constraint = False
            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)

                # TODO make allowed_mask symmetric
                import ipdb
                ipdb.set_trace()

                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the approximate meta gradients.
            adj_meta_approx_argmax = torch.argmax(adj_meta_grad)

            row_idx, col_idx = utils.unravel_index(adj_meta_approx_argmax, ori_adj.shape)

            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                self.perturb_features()

        return self.adj_changes + ori_adj

    def perturb_features(self):

        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes)[0]
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        adj_meta_grad -= adj_meta_grad.min()
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        feature_grad = torch.autograd.grad(attack_loss, self.feature_changes)[0]
        feature_meta_grad = feature_grad * (-2 * feature + 1)
        # TODO
        feature_meta_grad -= feature_meta_grad.min()
        features_meta_grad_argmax = torch.argmax(features_meta_grad)

        # Get meta gradients of the attributes.
        self.attribute_meta_grad = tf.multiply(tf.gradients(attack_loss, self.attribute_changes)[0],
                                               tf.reshape(self.attributes, [-1]) * -2 + 1)
        self.attribute_meta_grad -= tf.reduce_min(self.attribute_meta_grad)

        attribute_meta_grad_argmax = tf.argmax(self.attribute_meta_grad)

        self.attribute_meta_update = tf.scatter_add(self.attribute_changes,
                                                    indices=attribute_meta_grad_argmax,
                                                    updates=-2 * tf.gather(
                                                        tf.reshape(self.attributes, [-1]),
                                                        attribute_meta_grad_argmax) + 1),

        adjacency_meta_grad_max = tf.reduce_max(self.adjacency_meta_grad)
        attribute_meta_grad_max = tf.reduce_max(self.attribute_meta_grad)

        cond = adjacency_meta_grad_max > attribute_meta_grad_max

        self.combined_update = tf.cond(cond, lambda: self.adjacency_meta_update,
                                       lambda: self.attribute_meta_update)

class MetaApprox(BaseMeta):

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu', lambda_=0.5, train_iters=100, lr=0.01):

        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_features, lambda_, device)

        self.lr = lr
        self.train_iters = train_iters

        self.adj_meta_grad = None
        self.features_meta_grad = None

        self.grad_sum = torch.zeros(nnodes, nnodes).to(device)

        self.weights = []
        self.biases = []
        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr) # , weight_decay=5e-4)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def single_inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        for j in range(self.train_iters):
            self.optimizer.zero_grad()
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix]*float(self.with_bias)
                if ix == 0 and self.sparse_features:
                    hidden = adj_norm @ torch.spmm(features, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
            loss_labeled.backward()
            self.optimizer.step()

        return loss_labeled, loss_unlabeled

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            hidden = features
            for w, b in zip(self.weights, self.biases):
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

            self.grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)
            self.optimizer.step()

        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print(f'GCN loss on unlabled data: {loss_test_val.item()}')
        print(f'GCN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')

        # self.adj_changes.grad.zero_()

    def attack(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        labels_self_training = self.train_surrogate(features, ori_adj, labels, idx_train)
        self.sparse_features = sp.issparse(features)

        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            self.grad_sum.data.fill_(0)
            self.inner_train(features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_grad = self.grad_sum * (-2 * modified_adj + 1)
            # Make sure that the minimum entry is 0.
            adj_meta_grad -= adj_meta_grad.min()

            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                # Set entries to 0 that would violate the log likelihood constraint.
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the approximate meta gradients.
            adj_meta_approx_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = utils.unravel_index(adj_meta_approx_argmax, ori_adj.shape)

            # self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            # self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if modifed_adj[row_idx][col_idx] == 0:
                self.adj_changes.data[row_idx][col_idx] = 1
                self.adj_changes.data[col_idx][row_idx] = 1
            else:
                self.adj_changes.data[row_idx][col_idx] = 0
                self.adj_changes.data[col_idx][row_idx] = 0

        return self.adj_changes + ori_adj



