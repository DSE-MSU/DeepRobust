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
from deeprobust.graph.rl.nipa_q_net_node import QNetNode, NStepQNetNode, node_greedy_actions
from deeprobust.graph.rl.nstep_replay_mem import NstepReplayMem
from deeprobust.graph.utils import loss_acc

class NIPA(object):

    def __init__(self, env, features, labels, idx_train, idx_test,
            list_action_space, ratio, reward_type='binary', batch_size=20,
            num_wrong=0, bilin_q=1, embed_dim=64, gm='mean_field',
            mlp_hidden=64, max_lv=1, save_dir='checkpoint_dqn', device=None):

        assert device is not None, "'device' cannot be None, please specify it"

        self.features = features
        self.labels = labels
        self.possible_labels = torch.arange(labels.max() + 1).to(labels.device)
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.num_wrong = num_wrong
        self.list_action_space = list_action_space

        degrees = np.array([len(d) for n, d in list_action_space.items()])
        N = len(degrees[degrees > 0])
        self.n_injected = len(degrees) - N
        assert self.n_injected == int(ratio * N)
        self.injected_nodes = np.arange(N)[-self.n_injected: ]

        self.reward_type = reward_type
        self.batch_size = batch_size
        self.save_dir = save_dir
        if not osp.exists(save_dir):
            os.system(f'mkdir -p {save_dir}')

        self.gm = gm
        self.device = device

        self.mem_pool = NstepReplayMem(memory_size=500000, n_steps=3, balance_sample=reward_type == 'binary', model='nipa')
        self.env = env

        # self.net = QNetNode(features, labels, list_action_space)
        # self.old_net = QNetNode(features, labels, list_action_space)
        self.net = NStepQNetNode(3, features, labels, list_action_space, self.n_injected,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.old_net = NStepQNetNode(3, features, labels, list_action_space, self.n_injected,
                          bilin_q=bilin_q, embed_dim=embed_dim, mlp_hidden=mlp_hidden,
                          max_lv=max_lv, gm=gm, device=device)

        self.net = self.net.to(device)
        self.old_net = self.old_net.to(device)

        self.eps_start = 1.0
        self.eps_end = 0.05
        # self.eps_step = 100000
        self.eps_step = 10000
        self.burn_in = 50
        self.step = 0
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        # TODO
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:

            cur_state = self.env.getStateRef()
            list_at = self.env.uniformRandActions()
            actions = self.possible_actions(cur_state, list_at, time_t)
            actions, values = self.net(time_t, cur_state, actions, greedy_acts=True, is_inference=True)

            assert len(actions) == len(list_at)
            # actions = list(actions.cpu().numpy())

        return actions

    def run_whole_simulation(self):
        self.env.init_overall_steps()
        while not self.env.isTerminal():
            self.run_simulation()

    def run_simulation(self):
        self.env.setup()
        t = 0
        while not self.env.isActionFinished():
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()

            self.env.step(list_at)

            assert (self.env.rewards is not None) == self.env.isActionFinished()
            if self.env.isActionFinished():
                rewards = self.env.rewards
                s_prime = self.env.cloneState()
            else:
                # rewards = env.rewards
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            if self.env.isTerminal():
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = None
                self.env.init_overall_steps()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime,
                                    [self.env.isTerminal()] * len(list_at), t)
            t += 1

    def eval(self, training=True):
        self.env.init_overall_steps()
        self.env.setup()

        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t % 3, greedy=True)
            self.env.step(list_at)
            t += 1

        device = self.labels.device
        extra_adj = self.env.modified_list[0].get_extra_adj(device=device)
        adj = self.env.classifier.norm_tool.norm_extra(extra_adj)
        labels = torch.cat((self.labels, self.env.modified_label_list[0]))
        # self.classifier.fit(self.features, adj, labels, self.idx_train, self.idx_val, normalize=False)
        self.env.classifier.fit(self.features, adj, labels, self.idx_train, normalize=False)
        output = self.env.classifier(self.features, adj)
        loss, acc = loss_acc(output, self.labels, self.idx_test)
        print('\033[93m average test: acc %.5f\033[0m' % (acc))

        if training == True and self.best_eval is None or acc < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), osp.join(self.save_dir, 'epoch-best.model'))
            with open(osp.join(self.save_dir, 'epoch-best.txt'), 'w') as f:
                f.write('%.4f\n' % acc)
            # with open(osp.join(self.save_dir, 'attack_solution.txt'), 'w') as f:
            #     for i in range(len(self.idx_meta)):
            #         f.write('%d: [' % self.idx_meta[i])
            #         for e in self.env.modified_list[i].directed_edges:
            #             f.write('(%d %d)' % e)
            #         f.write('] succ: %d\n' % (self.env.binary_rewards[i]))
            self.best_eval = acc

    def train(self, episodes=10, num_steps=100000, lr=0.01):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.env.init_overall_steps()
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()

        self.mem_pool.print_count()

        self.env.init_overall_steps()
        pbar = tqdm(range(num_steps), unit='steps')
        for self.step in pbar:
            self.run_simulation()
            if self.step % 123 == 0:
                self.take_snapshot()

            if self.step % 1000 == 0:
                self.eval()

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt).to(self.device)

            if not list_term[0]:
                # target_nodes, _, picked_nodes = zip(*list_s_primes)
                actions = self.possible_actions(list_st, list_at, cur_time+1)
                _, q_rhs = self.old_net(cur_time + 1, list_s_primes, actions, greedy_acts=True)
                list_target += q_rhs

            # list_target = list_target.view(-1, 1)
            _, q_sa = self.net(cur_time, list_st, list_at)
            # q_sa = torch.cat(q_sa, dim=0)
            loss = F.mse_loss(q_sa, list_target)
            loss = torch.clamp(loss, -1, 1)
            optimizer.zero_grad()
            # print([x[0] for x in self.named_parameters() if x[1].grad is None])
            loss.backward()
            optimizer.step()
            pbar.set_description('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )
            # print('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )

    def possible_actions(self, list_st, list_at, t):
        '''
            list_st: current state
            list_at: current action
            return: actions for next state
        '''
        t = t % 3
        if t == 0:
            return np.tile(self.injected_nodes, ((len(list_at), 1)))

        if t == 1:
            actions = []
            for i in range(len(list_at)):
                a_prime = list_st[i][0].get_possible_nodes(list_at[0])
                actions.append(a_prime)
            return actions

        if t == 2:
            return self.possible_labels.repeat((len(list_at), 1))

