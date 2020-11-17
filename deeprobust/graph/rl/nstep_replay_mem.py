'''
    This part of code is adopted from https://github.com/Hanjun-Dai/graph_adversarial_attack (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le)
    but modified to be integrated into the repository.
'''

import random
import numpy as np

class NstepReplaySubMemCell(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.actions = [None] * self.memory_size
        self.rewards = [None] * self.memory_size
        self.states = [None] * self.memory_size
        self.s_primes = [None] * self.memory_size
        self.terminals = [None] * self.memory_size

        self.count = 0
        self.current = 0

    def add(self, s_t, a_t, r_t, s_prime, terminal):
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i])

    def sample(self, batch_size):

        assert self.count >= batch_size
        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []

        for i in range(batch_size):
            idx = random.randint(0, self.count - 1)
            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(float(self.rewards[idx]))
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])

        return list_st, list_at, list_rt, list_s_primes, list_term

def hash_state_action(s_t, a_t):
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base

    key = (key * base + a_t) % base
    return key

def nipa_hash_state_action(s_t, a_t):
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base

    key = (key * base + a_t) % base
    return key

class NstepReplayMemCell(object):
    def __init__(self, memory_size, balance_sample = False):
        self.sub_list = []
        self.balance_sample = balance_sample
        self.sub_list.append(NstepReplaySubMemCell(memory_size))
        if balance_sample:
            self.sub_list.append(NstepReplaySubMemCell(memory_size))
            self.state_set = set()

    def add(self, s_t, a_t, r_t, s_prime, terminal, use_hash=True):
        if not self.balance_sample or r_t < 0:
            self.sub_list[0].add(s_t, a_t, r_t, s_prime, terminal)
        else:
            assert r_t > 0
            if use_hash:
                # TODO add hash?
                key = hash_state_action(s_t, a_t)
                if key in self.state_set:
                    return
                self.state_set.add(key)
            self.sub_list[1].add(s_t, a_t, r_t, s_prime, terminal)

    def sample(self, batch_size):
        if not self.balance_sample or self.sub_list[1].count < batch_size:
            return self.sub_list[0].sample(batch_size)

        list_st, list_at, list_rt, list_s_primes, list_term = self.sub_list[0].sample(batch_size // 2)
        list_st2, list_at2, list_rt2, list_s_primes2, list_term2 = self.sub_list[1].sample(batch_size - batch_size // 2)

        return list_st + list_st2, list_at + list_at2, list_rt + list_rt2, list_s_primes + list_s_primes2, list_term + list_term2

class NstepReplayMem(object):
    def __init__(self, memory_size, n_steps, balance_sample=False, model='rl_s2v'):
        self.mem_cells = []
        for i in range(n_steps - 1):
            self.mem_cells.append(NstepReplayMemCell(memory_size, False))
        self.mem_cells.append(NstepReplayMemCell(memory_size, balance_sample))

        self.n_steps = n_steps
        self.memory_size = memory_size
        self.model = model

    def add(self, s_t, a_t, r_t, s_prime, terminal, t):
        assert t >= 0 and t < self.n_steps
        if self.model == 'nipa':
            self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal, use_hash=False)
        else:
            if t == self.n_steps - 1:
                assert terminal
            else:
                assert not terminal
            self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal, use_hash=True)

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term, t):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i], t)

    def sample(self, batch_size, t = None):
        if t is None:
            t = np.random.randint(self.n_steps)
            list_st, list_at, list_rt, list_s_primes, list_term = self.mem_cells[t].sample(batch_size)
        return t, list_st, list_at, list_rt, list_s_primes, list_term

    def print_count(self):
        for i in range(self.n_steps):
            for j, cell in enumerate(self.mem_cells[i].sub_list):
                print('Cell {} sub_list {}: {}'.format(i, j, cell.count))
