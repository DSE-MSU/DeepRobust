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

    def __init__(self, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        '''
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        '''
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm

        if bilin_q:
            last_wout = embed_dim
        else:
            last_wout = 1
            self.bias_target = Parameter(torch.Tensor(1, embed_dim))

        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)

        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)

        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp.cuda()
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):

        if self.node_features.data.is_sparse:
            input_node_linear = torch.spmm(self.node_features, self.w_n2l)
        else:
            input_node_linear = torch.mm(self.node_features, self.w_n2l)

        input_node_linear += self.bias_n2l

        # TODO the number of target nodes is batch_size, it actually parallizes
        target_nodes, batch_graph, picked_nodes = zip(*states)

        list_pred = []
        prefix_sum = []
        for i in range(len(batch_graph)):
            region = self.list_action_space[target_nodes[i]]

            node_embed = input_node_linear.clone()
            if picked_nodes is not None and picked_nodes[i] is not None:
                with torch.set_grad_enabled(mode=not is_inference):
                    picked_sp =  self.make_spmat(self.total_nodes, 1, picked_nodes[i], 0)
                    node_embed += torch.spmm(picked_sp, self.bias_picked)
                    region = self.list_action_space[picked_nodes[i]]

            if not self.bilin_q:
                with torch.set_grad_enabled(mode=not is_inference):
                # with torch.no_grad():
                    target_sp = self.make_spmat(self.total_nodes, 1, target_nodes[i], 0)
                    node_embed += torch.spmm(target_sp, self.bias_target)

            with torch.set_grad_enabled(mode=not is_inference):
                device = self.node_features.device
                adj = self.norm_tool.norm_extra( batch_graph[i].get_extra_adj(device))

                lv = 0
                input_message = node_embed

                node_embed = F.relu(input_message)
                while lv < self.max_lv:
                    n2npool = torch.spmm(adj, node_embed)
                    node_linear = self.conv_params( n2npool )
                    merged_linear = node_linear + input_message
                    node_embed = F.relu(merged_linear)
                    lv += 1

                target_embed = node_embed[target_nodes[i], :].view(-1, 1)
                if region is not None:
                    node_embed = node_embed[region]

                graph_embed = torch.mean(node_embed, dim=0, keepdim=True)

                if actions is None:
                    graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
                else:
                    if region is not None:
                        act_idx = region.index(actions[i])
                    else:
                        act_idx = actions[i]
                    node_embed = node_embed[act_idx, :].view(1, -1)

                embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
                if self.mlp_hidden:
                    embed_s_a = F.relu( self.linear_1(embed_s_a) )
                raw_pred = self.linear_out(embed_s_a)

                if self.bilin_q:
                    raw_pred = torch.mm(raw_pred, target_embed)
                list_pred.append(raw_pred)

        if greedy_acts:
            actions, _ = node_greedy_actions(target_nodes, picked_nodes, list_pred, self)

        return actions, list_pred

class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):

        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        list_mod = []
        for i in range(0, num_steps):
            # list_mod.append(QNetNode(node_features, node_labels, list_action_space))
            list_mod.append(QNetNode(node_features, node_labels, list_action_space, bilin_q, embed_dim, mlp_hidden, max_lv, gm=gm, device=device))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps

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


