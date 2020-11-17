"""
    Adversarial Attacks on Neural Networks for Graph Data. ICML 2018.
        https://arxiv.org/abs/1806.02371
    Author's Implementation
       https://github.com/Hanjun-Dai/graph_adversarial_attack
    This part of code is adopted from the author's implementation (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le) but modified
    to be integrated into the repository.
"""

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import pickle as cp
from deeprobust.graph.utils import *
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from deeprobust.graph import utils

class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size( (len(StaticGraph.graph), len(StaticGraph.graph)) )

class GraphNormTool(object):

    def __init__(self, normalize, gm, device):
        self.adj_norm = normalize
        self.gm = gm
        g = StaticGraph.graph
        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)

        # self_edges = np.array([range(len(g)), range(len(g))], dtype=np.int64)
        # edges = np.hstack((edges.T, rev_edges, self_edges))
        edges = np.hstack((edges.T, rev_edges))
        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])

        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        self.raw_adj = self.raw_adj.to(device)

        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                self.normed_adj = utils.normalize_adj_tensor(self.normed_adj, sparse=True)
                # GraphLaplacianNorm(self.normed_adj)
            else:

                self.normed_adj = utils.degree_normalize_adj_tensor(self.normed_adj, sparse=True)
                # GraphDegreeNorm(self.normed_adj)

    def norm_extra(self, added_adj = None):
        if added_adj is None:
            return self.normed_adj

        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse=True)

        return new_adj


class ModifiedGraph(object):
    def __init__(self, directed_edges = None, weights = None):
        self.edge_set = set()  #(first, second)
        self.node_set = set(range(StaticGraph.get_gsize()[0]))
        self.node_set = np.arange(StaticGraph.get_gsize()[0])
        if directed_edges is not None:
            self.directed_edges = deepcopy(directed_edges)
            self.weights = deepcopy(weights)
        else:
            self.directed_edges = []
            self.weights = []

    def add_edge(self, x, y, z):
        assert x is not None and y is not None
        if x == y:
            return
        for e in self.directed_edges:
            if e[0] == x and e[1] == y:
                return
            if e[1] == x and e[0] == y:
                return
        self.edge_set.add((x, y)) # (first, second)
        self.edge_set.add((y, x)) # (second, first)
        self.directed_edges.append((x, y))
        # assert z < 0
        self.weights.append(z)

    def get_extra_adj(self, device):
        if len(self.directed_edges):
            edges = np.array(self.directed_edges, dtype=np.int64)
            rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
            edges = np.hstack((edges.T, rev_edges))

            idxes = torch.LongTensor(edges)
            values = torch.Tensor(self.weights + self.weights)

            added_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())

            added_adj = added_adj.to(device)
            return added_adj
        else:
            return None

    def get_possible_nodes(self, target_node):
        connected = set()
        connected = []
        for n1, n2 in self.edge_set:
            if n1 == target_node:
                # connected.add(target_node)
                connected.append(n1)
        return np.setdiff1d(self.node_set, np.array(connected))
        # return self.node_set - connected

class NodeAttackEnv(object):
    """Node attack environment. It executes an action and then change the
    environment status (modify the graph).
    """

    def __init__(self, features, labels, all_targets, list_action_space, classifier, num_mod=1, reward_type='binary'):

        self.classifier = classifier
        self.list_action_space = list_action_space
        self.features = features
        self.labels = labels
        self.all_targets = all_targets
        self.num_mod = num_mod
        self.reward_type = reward_type

    def setup(self, target_nodes):
        self.target_nodes = target_nodes
        self.n_steps = 0
        self.first_nodes = None
        self.rewards = None
        self.binary_rewards = None
        self.modified_list = []
        for i in range(len(self.target_nodes)):
            self.modified_list.append(ModifiedGraph())

        self.list_acc_of_all = []

    def step(self, actions):
        """run actions and get rewards
        """
        if self.first_nodes is None: # pick the first node of edge
            assert self.n_steps % 2 == 0
            self.first_nodes = actions[:]
        else:
            for i in range(len(self.target_nodes)):
                # assert self.first_nodes[i] != actions[i]
                # deleta an edge from the graph
                self.modified_list[i].add_edge(self.first_nodes[i], actions[i], -1.0)
            self.first_nodes = None
            self.banned_list = None
        self.n_steps += 1

        if self.isTerminal():
            # only calc reward when its terminal
            acc_list = []
            loss_list = []
            # for i in tqdm(range(len(self.target_nodes))):
            for i in (range(len(self.target_nodes))):
                device = self.labels.device
                extra_adj = self.modified_list[i].get_extra_adj(device=device)
                adj = self.classifier.norm_tool.norm_extra(extra_adj)

                output = self.classifier(self.features, adj)

                loss, acc = loss_acc(output, self.labels, self.all_targets, avg_loss=False)
                # _, loss, acc = self.classifier(self.features, Variable(adj), self.all_targets, self.labels, avg_loss=False)

                cur_idx = self.all_targets.index(self.target_nodes[i])
                acc = np.copy(acc.double().cpu().view(-1).numpy())
                loss = loss.data.cpu().view(-1).numpy()
                self.list_acc_of_all.append(acc)
                acc_list.append(acc[cur_idx])
                loss_list.append(loss[cur_idx])

            self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            if self.reward_type == 'binary':
                self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            else:
                assert self.reward_type == 'nll'
                self.rewards = np.array(loss_list).astype(np.float32)

    def sample_pos_rewards(self, num_samples):
        assert self.list_acc_of_all is not None
        cands = []

        for i in range(len(self.list_acc_of_all)):
            succ = np.where( self.list_acc_of_all[i] < 0.9 )[0]

            for j in range(len(succ)):

                cands.append((i, self.all_targets[succ[j]]))

        if num_samples > len(cands):
            return cands
        random.shuffle(cands)
        return cands[0:num_samples]

    def uniformRandActions(self):
        # TODO: here only support deleting edges
        # seems they sample first node from 2-hop neighbours
        act_list = []
        offset = 0
        for i in range(len(self.target_nodes)):
            cur_node = self.target_nodes[i]
            region = self.list_action_space[cur_node]

            if self.first_nodes is not None and self.first_nodes[i] is not None:
                region = self.list_action_space[self.first_nodes[i]]

            if region is None:  # singleton node
                cur_action = np.random.randint(len(self.list_action_space))
            else: # select from neighbours or 2-hop neighbours
                cur_action = region[np.random.randint(len(region))]

            act_list.append(cur_action)
        return act_list

    def isTerminal(self):
        if self.n_steps == 2 * self.num_mod:
            return True
        return False

    def getStateRef(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes

        return zip(self.target_nodes, self.modified_list, cp_first)

    def cloneState(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]

        return list(zip(self.target_nodes[:], deepcopy(self.modified_list), cp_first))


