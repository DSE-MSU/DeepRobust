'''
    Adversarial Attacks on Neural Networks for Graph Data. ICML 2018.
        https://arxiv.org/abs/1806.02371
    Author's Implementation
       https://github.com/Hanjun-Dai/graph_adversarial_attack
    This part of code is adopted from the author's implementation (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le) but modified
    to be integrated into the repository.
'''

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
from deeprobust.graph.rl.env import GraphNormTool

class QNetNode(nn.Module):

    def __init__(self, node_features, node_labels, list_action_space, n_injected, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        '''
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        '''
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.identity = torch.eye(node_labels.max() + 1).to(node_labels.device)
        # self.node_labels = self.to_onehot(node_labels)
        self.n_injected = n_injected

        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm

        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 3, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, 1)
        else:
            self.linear_out = nn.Linear(embed_dim * 3, 1)

        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))

        # self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        weights_init(self)

        input_dim = (node_labels.max() + 1) * self.n_injected
        self.label_encoder_1 = nn.Linear(input_dim, mlp_hidden)
        self.label_encoder_2 = nn.Linear(mlp_hidden, embed_dim)
        self.device = self.node_features.device

    def to_onehot(self, labels):
        return self.identity[labels].view(-1, self.identity.shape[1])

    def get_label_embedding(self, labels):
        # int to one hot
        onehot = self.to_onehot(labels).view(1, -1)

        x = F.relu(self.label_encoder_1(onehot))
        x = F.relu(self.label_encoder_2(x))
        return x

    def get_action_label_encoding(self, label):
        onehot = self.to_onehot(label)
        zeros = torch.zeros((onehot.shape[0], self.embed_dim - onehot.shape[1])).to(onehot.device)
        return torch.cat((onehot, zeros), dim=1)

    def get_graph_embedding(self, adj):
        if self.node_features.data.is_sparse:
            node_embed = torch.spmm(self.node_features, self.w_n2l)
        else:
            node_embed = torch.mm(self.node_features, self.w_n2l)

        node_embed += self.bias_n2l

        input_message = node_embed
        node_embed = F.relu(input_message)

        for i in range(self.max_lv):
            n2npool = torch.spmm(adj, node_embed)
            node_linear = self.conv_params(n2npool)
            merged_linear = node_linear + input_message
            node_embed = F.relu(merged_linear)

        graph_embed = torch.mean(node_embed, dim=0, keepdim=True)
        return graph_embed, node_embed

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)

        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp.cuda()
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):

        preds = torch.zeros(len(states)).to(self.device)

        batch_graph, modified_labels = zip(*states)
        greedy_actions = []
        with torch.set_grad_enabled(mode=not is_inference):

            for i in range(len(batch_graph)):
                if batch_graph[i] is None:
                    continue
                adj = self.norm_tool.norm_extra(batch_graph[i].get_extra_adj(self.device))
                # get graph representation
                graph_embed, node_embed = self.get_graph_embedding(adj)

                # get label reprensentation
                label_embed = self.get_label_embedding(modified_labels[i])

                # get action reprensentation
                if time_t != 2:
                    action_embed = node_embed[actions[i]].view(-1, self.embed_dim)
                else:
                    action_embed = self.get_action_label_encoding(actions[i])

                # concat them and send it to neural network
                embed_s = torch.cat((graph_embed, label_embed), dim=1)
                embed_s = embed_s.repeat(len(action_embed), 1)
                embed_s_a = torch.cat((embed_s, action_embed), dim=1)

                if self.mlp_hidden:
                    embed_s_a = F.relu( self.linear_1(embed_s_a) )

                raw_pred = self.linear_out(embed_s_a)

                if greedy_acts:
                    action_id = raw_pred.argmax(0)
                    raw_pred = raw_pred.max()
                    greedy_actions.append(actions[i][action_id])
                else:
                    raw_pred = raw_pred.max()
                # list_pred.append(raw_pred)
                preds[i] += raw_pred


        return greedy_actions, preds

class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels, list_action_space, n_injected, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):

        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        list_mod = []
        for i in range(0, num_steps):
            # list_mod.append(QNetNode(node_features, node_labels, list_action_space))
            list_mod.append(QNetNode(node_features, node_labels, list_action_space, n_injected, bilin_q, embed_dim, mlp_hidden, max_lv, gm=gm, device=device))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False, is_inference=False):
        # print('time_t:', time_t)
        # print('self.num_step:', self.num_steps)
        # assert time_t >= 0 and time_t < self.num_steps
        time_t = time_t % 3
        return self.list_mod[time_t](time_t, states, actions, greedy_acts, is_inference)


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

def node_greedy_actions(target_nodes, picked_nodes, list_q, net):
    assert len(target_nodes) == len(list_q)

    actions = []
    values = []
    for i in range(len(target_nodes)):
        region = net.list_action_space[target_nodes[i]]
        if picked_nodes is not None and picked_nodes[i] is not None:
            region = net.list_action_space[picked_nodes[i]]
        if region is None:
            assert list_q[i].size()[0] == net.total_nodes
        else:
            assert len(region) == list_q[i].size()[0]

        val, act = torch.max(list_q[i], dim=0)
        values.append(val)
        if region is not None:
            act = region[act.data.cpu().numpy()[0]]
            # act = Variable(torch.LongTensor([act]))
            act = torch.LongTensor([act])
            actions.append(act)
        else:
            actions.append(act)

    return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data


