import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from DeepRobust.graph import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Random(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, add_nodes=False, device='cpu'):
        """
        """
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, adj, n_perturbations, filp_edge=False):
        pass

    def attack_structure(self, adj, n_perturbations, filp_edges=False):
        """
        Randomly add or flip edges.
        """
        # adj: sp.csr_matrix

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = adj.tolil()

        # sample edges to add/flip edges
        edges = modified_adj.nonzero().T
        import ipdb
        ipdb.set_trace()
        edges = np.random.permutation(edges)[: n_perturbations]
        edges
        return modified_adj

    def attack_structure(self, features, n_perturbations):
        """
        Randomly perturb features.
        """
        print(f'number of pertubations: {n_perturbations}')

        return modified_features

    def add_nodes(self, adj, added_nodes, n_perturbations):
        """
        For each added node, randomly connect with other nodes.
        """
        # adj: sp.csr_matrix
        print(f'number of pertubations: {n_perturbations}')
        modified_adj = adj.tolil()
        return modified_adj

