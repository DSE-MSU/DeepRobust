import torch
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from deeprobust.graph import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp

class RND(BaseAttack):
    """As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
    'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
    in each step we randomly sample nodes u whose lable is different from v and
    add the edge u,v to the graph structure

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.targeted_attack import RND
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = RND()
    >>> # Attack
    >>> model.attack(adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> # # You can also inject nodes
    >>> # model.add_nodes(features, adj, labels, idx_train, target_node, n_added=10, n_perturbations=100)
    >>> # modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'RND does NOT support attacking features except adding nodes'

    def attack(self, ori_adj, labels, idx_train, target_node, n_perturbations, **kwargs):
        """
        Randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """
        # ori_adj: sp.csr_matrix

        print('number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()

        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] \
                            and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes,
                            unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        # self.modified_features = modified_features

    def add_nodes(self, features, ori_adj, labels, idx_train, target_node, n_added=1, n_perturbations=10, **kwargs):
        """
        For each added node, first connect the target node with added fake nodes.
        Then randomly connect the fake nodes with other nodes whose label is
        different from target node. As for the node feature, simply copy arbitary node
        """
        # ori_adj: sp.csr_matrix
        print('number of pertubations: %s' % n_perturbations)
        N = ori_adj.shape[0]
        D = features.shape[1]
        modified_adj = self.reshape_mx(ori_adj, shape=(N+n_added, N+n_added))
        modified_features = self.reshape_mx(features, shape=(N+n_added, D))

        diff_labels = [l for l in range(labels.max()+1) if l != labels[target_node]]
        diff_labels = np.random.permutation(diff_labels)
        possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]

        for fake_node in range(N, N+n_added):
            sampled_nodes = np.random.permutation(possible_nodes)[: n_perturbations]
            # connect the fake node with target node
            modified_adj[fake_node, target_node] = 1
            modified_adj[target_node, fake_node] = 1
            # connect the fake node with other nodes
            for node in sampled_nodes:
                modified_adj[fake_node, node] = 1
                modified_adj[node, fake_node] = 1
            modified_features[fake_node] = features[node]

        self.check_adj(modified_adj)

        self.modified_adj = modified_adj
        self.modified_features = modified_features
        # return modified_adj, modified_features

    def reshape_mx(self, mx, shape):
        indices = mx.nonzero()
        return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape).tolil()

