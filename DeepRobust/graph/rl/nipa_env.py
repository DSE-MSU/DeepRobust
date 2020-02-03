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
from DeepRobust.graph.utils import *
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from DeepRobust.graph import utils
from DeepRobust.graph.rl.env import *

class NodeInjectionEnv(NodeAttackEnv):

    def __init__(self, features, labels, idx_val, dict_of_lists, classifier, ratio=0.01, parallel_size=2, reward_type='binary'):
        '''
            number of injected nodes: ratio*|V|
            number of modifications: ratio*|V|*|D_avg|
        '''
        # super(NodeInjectionEnv, self).__init__(features, labels, all_targets, list_action_space, classifier, num_mod, reward_type)
        super(NodeInjectionEnv, self).__init__(features, labels, idx_val, dict_of_lists, classifier)

        self.parallel_size = parallel_size

        degrees = np.array([len(d) for n, d in dict_of_lists.items()])
        N = len(degrees[degrees > 0])
        avg_degree = degrees.sum() / N
        self.n_injected = len(degrees) - N
        assert self.n_injected == int(ratio * N)

        self.n_perturbations = int(self.n_injected * avg_degree)
        self.all_nodes = np.arange(N)
        self.injected_nodes = self.all_nodes[-self.n_injected: ]
        self.previous_acc = [1] * parallel_size
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

    def step(self, actions):
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
                import ipdb
                ipdb.set_trace()
                self.modified_label_list[i][self.first_nodes[i]] = actions[i]

            self.first_nodes = None
            self.second_nodes = None

        self.n_steps += 1
        self.overall_steps += 1

        if self.isActionFinished():
            acc_list = []
            loss_list = []
            rewards = []
            for i in (range(self.parallel_size)):
                device = self.labels.device
                extra_adj = self.modified_list[i].get_extra_adj(device=device)
                adj = self.classifier.norm_tool.norm_extra(extra_adj)
                output = self.classifier(self.features, adj)
                loss, acc = loss_acc(output, self.labels, self.idx_val)

                r = 1 if self.previous_acc[i] > acc else -1
                rewards.append(r)

            self.rewards = np.array(rewards).astype(np.float32)

                # self.previous_acc[i] = acc
                # acc = np.copy(acc.double().cpu().view(-1).numpy())
                # loss = loss.data.cpu().view(-1).numpy()
                # self.list_acc_of_all.append(acc)
                # acc_list.append(acc)
                # loss_list.append(loss)

            # self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            # if self.reward_type == 'binary':
            #     self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)

        # if self.first_nodes is None: # pick the first node of edge
        #     assert self.n_steps % 2 == 0
        #     self.first_nodes = actions[:]
        # else:
        #     for i in range(len(self.target_nodes)):
        #         # assert self.first_nodes[i] != actions[i]
        #         # deleta an edge from the graph
        #         self.modified_list[i].add_edge(self.first_nodes[i], actions[i], -1.0)
        #     self.first_nodes = None
        #     self.banned_list = None
        # self.n_steps += 1

        # if self.isTerminal():
        #     # only calc reward when its terminal
        #     acc_list = []
        #     loss_list = []
        #     # for i in tqdm(range(len(self.target_nodes))):
        #     for i in (range(len(self.target_nodes))):
        #         device = self.labels.device
        #         extra_adj = self.modified_list[i].get_extra_adj(device=device)
        #         adj = self.classifier.norm_tool.norm_extra(extra_adj)

        #         output = self.classifier(self.features, adj)

        #         loss, acc = loss_acc(output, self.labels, self.all_targets, avg_loss=False)
        #         # _, loss, acc = self.classifier(self.features, Variable(adj), self.all_targets, self.labels, avg_loss=False)

        #         cur_idx = self.all_targets.index(self.target_nodes[i])
        #         acc = np.copy(acc.double().cpu().view(-1).numpy())
        #         loss = loss.data.cpu().view(-1).numpy()
        #         self.list_acc_of_all.append(acc)
        #         acc_list.append(acc[cur_idx])
        #         loss_list.append(loss[cur_idx])

        #     self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
        #     if self.reward_type == 'binary':
        #         self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
        #     else:
        #         assert self.reward_type == 'nll'
        #         self.rewards = np.array(loss_list).astype(np.float32)

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
                # TODO cannot be connected with a1
                cur_action = np.random.randint(len(self.list_action_space))
                while (self.first_nodes[i], cur_action) in self.modified_list[i].edge_set:

                    import ipdb
                    ipdb.set_trace()

                    cur_action = np.random.randint(len(self.list_action_space))

            if self.first_nodes is not None and self.second_nodes is not None:
                # a3: choose label
                cur_action = np.random.randint(self.label.max() + 1)
            act_list.append(cur_action)
        return act_list

    def isActionFinished(self):
        if (self.n_steps +1) % 3 == 0:
            return True
        return False

    def isTerminal(self):
        if self.overall_steps == self.n_perturbations:
            return True
        return False

    def getStateRef(self):
        # needn't copy

        import ipdb
        ipdb.set_trace()

        return list(zip(self.modified_list, self.modified_label_list))

        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes

        return zip(self.target_nodes, self.modified_list, cp_first)

    def cloneState(self):
        # TODO
        return list(zip(deepcopy(self.modified_list), deepcopy(self.labels[self.injected_nodes])))

        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]

        return list(zip(self.target_nodes[:], deepcopy(self.modified_list), cp_first))


