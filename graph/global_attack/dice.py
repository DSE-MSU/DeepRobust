import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from DeepRobust.graph import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp

class DICE(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        """
        As is described in ADVERSARIAL ATTACKS ON GRAPH NEURAL NETWORKS VIA META LEARNING (ICLR'19),
        'DICE (delete internally, connect externally) is a baseline where, for each perturbation,
        we randomly choose whether to insert or remove an edge. Edges are only removed between
        nodes from the same classes, and only inserted between nodes from different classes.
        """
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, adj, labels, n_perturbations):
        """
        Delete internally, connect externally. This baseline has all true class labels
        (train and test) available.
        """
        # adj: sp.csr_matrix
        # adj: tensor sparse?

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = deepcopy(adj).tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)
        n_insert = n_perturbations - n_remove
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]]== labels[x[1]]]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0

        for i in range(n_insert):
            # select a node
            node = np.random.randint(adj.shape[0])
            possible_nodes = [x for x in range(adj.shape[0])
                              if labels[x] != labels[node] and modified_adj[x, node] == 0]
            # select another node
            node2 = possible_nodes[np.random.randint(len(possible_nodes))]
            modified_adj[node, node2] = 1

        return modified_adj

