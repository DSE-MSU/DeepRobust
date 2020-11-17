'''
    This part of code is adopted from https://github.com/Hanjun-Dai/graph_adversarial_attack (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le)
    but modified to be integrated into the repository.
'''

import random
import numpy as np
from deeprobust.graph.rl.nstep_replay_mem import *


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


class NstepReplayMem(object):
    def __init__(self, memory_size, n_steps, balance_sample = False):
        self.mem_cells = []
        for i in range(n_steps - 1):
            self.mem_cells.append(NstepReplayMemCell(memory_size, False))
        self.mem_cells.append(NstepReplayMemCell(memory_size, balance_sample))

        self.n_steps = n_steps
        self.memory_size = memory_size

    def add(self, s_t, a_t, r_t, s_prime, terminal, t):
        assert t >= 0 and t < self.n_steps
        if t == self.n_steps - 1:
            assert terminal
        else:
            assert not terminal
        self.mem_cells[t].add(s_t, a_t, r_t, s_prime, terminal)

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
