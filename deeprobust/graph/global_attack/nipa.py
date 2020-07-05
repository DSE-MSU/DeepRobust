"""
Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach. WWW 2020.
https://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf

Still on testing stage. Haven't reproduced the performance yet.
"""

import os
import os.path as osp
import random
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from deeprobust.graph.rl.nipa_q_net_node import (NStepQNetNode, QNetNode,
                                                 node_greedy_actions)
from deeprobust.graph.rl.nstep_replay_mem import NstepReplayMem
from deeprobust.graph.utils import loss_acc


class NIPA(object):
    """ Reinforcement learning agent for NIPA attack.
    https://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf

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
    See more details in https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_nipa.py
    """

    def __init__(self, env, features, labels, idx_train, idx_val, idx_test,
            list_action_space, ratio, reward_type='binary', batch_size=30,
            num_wrong=0, bilin_q=1, embed_dim=64, gm='mean_field',
            mlp_hidden=64, max_lv=1, save_dir='checkpoint_dqn', device=None):

        assert device is not None, "'device' cannot be None, please specify it"

        self.features = features
        self.labels = labels
        self.possible_labels = torch.arange(labels.max() + 1).to(labels.device)
        self.idx_train = idx_train
        self.idx_val = idx_val
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
            os.system('mkdir -p %s' % save_dir)

        self.gm = gm
        self.device = device

        self.mem_pool = NstepReplayMem(memory_size=500000, n_steps=3, balance_sample=reward_type == 'binary', model='nipa')
        self.env = env

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
        self.eps_step = 30000
        self.GAMMA = 0.9
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

        self.step += 1
        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:

            cur_state = self.env.getStateRef()
            # list_at = self.env.uniformRandActions()
            list_at = self.env.first_nodes if time_t == 1 else None

            actions = self.possible_actions(cur_state, list_at, time_t)
            actions, values = self.net(time_t, cur_state, actions, greedy_acts=True, is_inference=True)

            assert len(actions) == len(cur_state)
            # actions = list(actions.cpu().numpy())
        return actions

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
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            if self.env.isTerminal():
                rewards = self.env.rewards
                s_prime = None
                # self.env.init_overall_steps()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime,
                                    [self.env.isTerminal()] * len(list_at), t)
            t += 1

    def eval(self, training=True):
        """Evaluate RL agent.
        """
        self.env.init_overall_steps()
        self.env.setup()

        for _ in count():
            self.env.setup()
            t = 0
            while not self.env.isActionFinished():
                list_at = self.make_actions(t, greedy=True)
                # print(list_at)
                self.env.step(list_at, inference=True)
                t += 1
            if self.env.isTerminal():
                break

        device = self.labels.device
        extra_adj = self.env.modified_list[0].get_extra_adj(device=device)
        adj = self.env.classifier.norm_tool.norm_extra(extra_adj)
        labels = torch.cat((self.labels, self.env.modified_label_list[0]))

        self.env.classifier.fit(self.features, adj, labels, self.idx_train, self.idx_val, normalize=False, patience=50)
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

    def train(self, num_episodes=10, lr=0.01):
        """Train RL agent.
        """
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.env.init_overall_steps()
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()
        self.mem_pool.print_count()

        for i_episode in tqdm(range(num_episodes)):
            self.env.init_overall_steps()

            for t in count():
                self.run_simulation()

                cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.batch_size)
                list_target = torch.Tensor(list_rt).to(self.device)

                if not list_term[0]:
                    actions = self.possible_actions(list_st, list_at, cur_time+1)
                    _, q_rhs = self.old_net(cur_time + 1, list_s_primes, actions, greedy_acts=True)
                    list_target += self.GAMMA * q_rhs

                # list_target = list_target.view(-1, 1)
                _, q_sa = self.net(cur_time, list_st, list_at)
                loss = F.mse_loss(q_sa, list_target)
                loss = torch.clamp(loss, -1, 1)
                optimizer.zero_grad()
                loss.backward()
                # print([x[0] for x in self.nnamed_parameters() if x[1].grad is None])
                # for param in self.net.parameters():
                #     if param.grad is None:
                #         continue
                #     param.grad.data.clamp_(-1, 1)
                optimizer.step()

                # pbar.set_description('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa)) )
                if t % 20 == 0:
                    print('eps: %.5f, loss: %0.5f, q_val: %.5f, list_target: %.5f' % (self.eps, loss, torch.mean(q_sa), torch.mean(list_target)) )

                if self.env.isTerminal():
                    break

                # if (t+1) % 50 == 0:
                #     self.take_snapshot()

            if i_episode % 1 == 0:
                self.take_snapshot()

            if i_episode % 1 == 0:
                self.eval()

    def possible_actions(self, list_st, list_at, t):
        """
        Parameters
        ----------
        list_st:
            current state
        list_at:
            current action

        Returns
        -------
        list
            actions for next state
        """

        t = t % 3
        if t == 0:
            return np.tile(self.injected_nodes, ((len(list_st), 1)))

        if t == 1:
            actions = []
            for i in range(len(list_at)):
                a_prime = list_st[i][0].get_possible_nodes(list_at[i])
                actions.append(a_prime)
            return actions

        if t == 2:
            return self.possible_labels.repeat((len(list_st), 1))
