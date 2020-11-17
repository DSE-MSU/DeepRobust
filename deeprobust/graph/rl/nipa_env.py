"""
    This part of code is adopted from https://github.com/Hanjun-Dai/graph_adversarial_attack (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le)
    but modified to be integrated into the repository.
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
from deeprobust.graph.rl.env import *

class NodeInjectionEnv(NodeAttackEnv):
    """Node attack environment. It executes an action and then change the
    environment status (modify the graph).
    """

    def __init__(self, features, labels, idx_train, idx_val, dict_of_lists, classifier, ratio=0.01, parallel_size=1, reward_type='binary'):
        """number of injected nodes: ratio*|V|
           number of modifications: ratio*|V|*|D_avg|
        """
        # super(NodeInjectionEnv, self).__init__(features, labels, all_targets, list_action_space, classifier, num_mod, reward_type)
        super(NodeInjectionEnv, self).__init__(features, labels, idx_val, dict_of_lists, classifier)
        self.parallel_size = parallel_size

        degrees = np.array([len(d) for n, d in dict_of_lists.items()])
        N = len(degrees[degrees > 0])
        avg_degree = degrees.sum() / N
        self.n_injected = len(degrees) - N
        assert self.n_injected == int(ratio * N)

        self.ori_adj_size = N
        self.n_perturbations = int(self.n_injected * avg_degree)
        print("number of perturbations: {}".format(self.n_perturbations))
        self.all_nodes = np.arange(N)
        self.injected_nodes = self.all_nodes[-self.n_injected: ]
        self.previous_acc = [1] * parallel_size

        self.idx_train = np.hstack((idx_train, self.injected_nodes))
        self.idx_val = idx_val

        self.modified_label_list = []
        for i in range(self.parallel_size):
            self.modified_label_list.append(labels[-self.n_injected: ].clone())


    def init_overall_steps(self):
        self.overall_steps = 0
        self.modified_list = []
        for i in range(self.parallel_size):
            self.modified_list.append(ModifiedGraph())

    def setup(self):
        self.n_steps = 0
        self.first_nodes = None
        self.second_nodes = None
        self.rewards = None
        self.binary_rewards = None
        self.list_acc_of_all = []

    def step(self, actions, inference=False):
        '''
            run actions and get reward
        '''
        if self.first_nodes is None: # pick the first node of edge
            assert (self.n_steps + 1) % 3 == 1
            self.first_nodes = actions[:]

        if (self.n_steps + 1) % 3 == 2:
            self.second_nodes = actions[:]
            for i in range(self.parallel_size):
                # add an edge from the graph
                self.modified_list[i].add_edge(self.first_nodes[i], actions[i], 1.0)

        if (self.n_steps + 1) % 3 == 0:
            for i in range(self.parallel_size):
                # change label
                self.modified_label_list[i][self.first_nodes[i] - self.ori_adj_size] = actions[i]

            self.first_nodes = None
            self.second_nodes = None

        self.n_steps += 1
        self.overall_steps += 1

        if not inference:
            if self.isActionFinished() :
                rewards = []
                for i in (range(self.parallel_size)):
                    device = self.labels.device
                    extra_adj = self.modified_list[i].get_extra_adj(device=device)
                    adj = self.classifier.norm_tool.norm_extra(extra_adj)
                    labels = torch.cat((self.labels, self.modified_label_list[i]))
                    # self.classifier.fit(self.features, adj, labels, self.idx_train, self.idx_val, normalize=False)
                    self.classifier.fit(self.features, adj, labels, self.idx_train, self.idx_val, normalize=False, patience=30)
                    output = self.classifier(self.features, adj)
                    loss, correct = loss_acc(output, self.labels, self.idx_val, avg_loss=False)
                    acc = correct.sum()
                    # r = 1 if self.previous_acc[i] - acc > 0.01  else -1
                    r = 1 if self.previous_acc[i] - acc > 0  else -1
                    self.previous_acc[i] = acc
                    rewards.append(r)
                    self.rewards = np.array(rewards).astype(np.float32)


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
        act_list = []
        for i in range(self.parallel_size):
            if self.first_nodes is None:
                # a1: choose a node from injected nodes
                cur_action = np.random.choice(self.injected_nodes)

            if self.first_nodes is not None and self.second_nodes is None:
                # a2: choose a node from all nodes
                cur_action = np.random.randint(len(self.list_action_space))
                while (self.first_nodes[i], cur_action) in self.modified_list[i].edge_set:
                    cur_action = np.random.randint(len(self.list_action_space))

            if self.first_nodes is not None and self.second_nodes is not None:
                # a3: choose label
                cur_action = np.random.randint(self.labels.cpu().max() + 1)

            act_list.append(cur_action)
        return act_list

    def isActionFinished(self):
        if (self.n_steps) % 3 == 0 and self.n_steps != 0:
            return True
        return False

    def isTerminal(self):
        if self.overall_steps == 3 * self.n_perturbations:
            return True
        return False

    def getStateRef(self):
        return list(zip(self.modified_list, self.modified_label_list))

    def cloneState(self):
        return list(zip(deepcopy(self.modified_list), deepcopy(self.modified_label_list)))

