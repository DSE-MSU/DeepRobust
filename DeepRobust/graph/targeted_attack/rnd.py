import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from DeepRobust.graph import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class RND(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        """
        As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
        'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
        in each step we randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure

        """
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, adj, labels, idx_train, target_node, n_perturbations):
        """
        Randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        """
        # adj: sp.csr_matrix

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = deepcopy(adj).tolil()

        row = adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] \
                            and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes,
                            unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1

        return modified_adj

