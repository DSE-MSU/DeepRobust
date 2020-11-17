"""
    Adversarial Attacks on Neural Networks for Graph Data. ICML 2018.
        https://arxiv.org/abs/1806.02371
    Author's Implementation
       https://github.com/Hanjun-Dai/graph_adversarial_attack
    This part of code is adopted from the author's implementation (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le)
    but modified to be integrated into the repository.
"""

import os
import sys
import os.path as osp
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
from deeprobust.graph.rl.q_net_node import QNetNode, NStepQNetNode, node_greedy_actions
from deeprobust.graph.rl.env import NodeAttackEnv
from deeprobust.graph.rl.nstep_replay_mem import NstepReplayMem

class RLS2V(object):
    """ Reinforcement learning agent for RL-S2V attack.

    Parameters
    ----------
    env :
        Node attack environment
    features :
        node features matrix
    labels :
        labels
    idx_meta :
        node meta indices
    idx_test :
        node test indices
    list_action_space : list
        list of action space
    num_mod :
        number of modification (perturbation) on the graph
    reward_type : str
        type of reward (e.g., 'binary')
    batch_size :
        batch size for training DQN
    save_dir :
        saving directory for model checkpoints
    device: str
        'cpu' or 'cuda'

    Examples
    --------
    See details in https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_rl_s2v.py
    """

    def __init__(self, env, features, labels, idx_meta, idx_test,
            list_action_space, num_mod, reward_type, batch_size=10,
            num_wrong=0, bilin_q=1, embed_dim=64, gm='mean_field',
            mlp_hidden=64, max_lv=1, save_dir='checkpoint_dqn', device=None):


        assert device is not None, "'device' cannot be None, please specify it"

        self.features = features
        self.labels = labels
        self.idx_meta = idx_meta
        self.idx_test = idx_test
        self.num_wrong = num_wrong
        self.list_action_space = list_action_space
        self.num_mod = num_mod
        self.reward_type = reward_type
        self.batch_size = batch_size
        self.save_dir = save_dir
        if not osp.exists(save_dir):
            os.system('mkdir -p {}'.format(save_dir))

        self.gm = gm
        self.device = device

        self.mem_pool = NstepReplayMem(memory_size=500000, n_steps=2 * num_mod, balance_sample=reward_type == 'binary')
        self.env = env

        # self.net = QNetNode(features, labels, list_action_space)
        # self.old_net = QNetNode(features, labels, list_action_space)
        self.net = NStepQNetNode(2 * num_mod, features, labels, list_action_space,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.old_net = NStepQNetNode(2 * num_mod, features, labels, list_action_space,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.net = self.net.to(device)
        self.old_net = self.old_net.to(device)

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = 10
        self.step = 0
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:
            cur_state = self.env.getStateRef()
            actions, values = self.net(time_t, cur_state, None, greedy_acts=True, is_inference=True)
            actions = list(actions.cpu().numpy())

        return actions

    def run_simulation(self):

        if (self.pos + 1) * self.batch_size > len(self.idx_test):
            self.pos = 0
            random.shuffle(self.idx_test)

        selected_idx = self.idx_test[self.pos * self.batch_size : (self.pos + 1) * self.batch_size]
        self.pos += 1
        self.env.setup(selected_idx)

        t = 0
        list_of_list_st = []
        list_of_list_at = []

        while not self.env.isTerminal():
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()

            self.env.step(list_at)

            # TODO Wei added line #87
            env = self.env
            assert (env.rewards is not None) == env.isTerminal()
            if env.isTerminal():
                rewards = env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t)
            list_of_list_st.append( deepcopy(list_st) )
            list_of_list_at.append( deepcopy(list_at) )
            t += 1

        # if the reward type is nll_loss, directly return
        if self.reward_type == 'nll':
            return

        T = t
        cands = self.env.sample_pos_rewards(len(selected_idx))
        if len(cands):
            for c in cands:
                sample_idx, target = c
                doable = True
                for t in range(T):
                    if self.list_action_space[target] is not None and (not list_of_list_at[t][sample_idx] in self.list_action_space[target]):
                        doable = False # TODO WHY False? This is only 1-hop neighbour
                        break
                if not doable:
                    continue

                for t in range(T):
                    s_t = list_of_list_st[t][sample_idx]
                    a_t = list_of_list_at[t][sample_idx]
                    s_t = [target, deepcopy(s_t[1]), s_t[2]]
                    if t + 1 == T:
                        s_prime = (None, None, None)
                        r = 1.0
                        term = True
                    else:
                        s_prime = list_of_list_st[t + 1][sample_idx]
                        s_prime = [target, deepcopy(s_prime[1]), s_prime[2]]
                        r = 0.0
                        term = False
                    self.mem_pool.mem_cells[t].add(s_t, a_t, r, s_prime, term)

    def eval(self, training=True):
        """Evaluate RL agent.
        """

        self.env.setup(self.idx_meta)
        t = 0

        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1

        acc = 1 - (self.env.binary_rewards + 1.0) / 2.0
        acc = np.sum(acc) / (len(self.idx_meta) + self.num_wrong)
        print('\033[93m average test: acc %.5f\033[0m' % (acc))

        if training == True and self.best_eval is None or acc < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), osp.join(self.save_dir, 'epoch-best.model'))
            with open(osp.join(self.save_dir, 'epoch-best.txt'), 'w') as f:
                f.write('%.4f\n' % acc)
            with open(osp.join(self.save_dir, 'attack_solution.txt'), 'w') as f:
                for i in range(len(self.idx_meta)):
                    f.write('%d: [' % self.idx_meta[i])
                    for e in self.env.modified_list[i].directed_edges:
                        f.write('(%d %d)' % e)
                    f.write('] succ: %d\n' % (self.env.binary_rewards[i]))
            self.best_eval = acc

    def train(self, num_steps=100000, lr=0.001):
        """Train RL agent.
        """

        pbar = tqdm(range(self.burn_in), unit='batch')

        for p in pbar:
            self.run_simulation()

        pbar = tqdm(range(num_steps), unit='steps')
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        for self.step in pbar:

            self.run_simulation()

            if self.step % 123 == 0:
                # update the params of old_net
                self.take_snapshot()
            if self.step % 500 == 0:
                self.eval()

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt).to(self.device)

            if not list_term[0]:
                target_nodes, _, picked_nodes = zip(*list_s_primes)
                _, q_t_plus_1 = self.old_net(cur_time + 1, list_s_primes, None)
                _, q_rhs = node_greedy_actions(target_nodes, picked_nodes, q_t_plus_1, self.old_net)
                list_target += q_rhs

            # list_target = Variable(list_target.view(-1, 1))
            list_target = list_target.view(-1, 1)
            _, q_sa = self.net(cur_time, list_st, list_at)
            q_sa = torch.cat(q_sa, dim=0)
            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )
            # print('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )

