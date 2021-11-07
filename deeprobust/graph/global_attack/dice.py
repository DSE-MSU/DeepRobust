import random
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.global_attack import BaseAttack

class DICE(BaseAttack):
    """As is described in ADVERSARIAL ATTACKS ON GRAPH NEURAL NETWORKS VIA META LEARNING (ICLR'19),
    'DICE (delete internally, connect externally) is a baseline where, for each perturbation,
    we randomly choose whether to insert or remove an edge. Edges are only removed between
    nodes from the same classes, and only inserted between nodes from different classes.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
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
    >>> from deeprobust.graph.global_attack import DICE
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = DICE()
    >>> model.attack(adj, labels, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_adj, labels, n_perturbations, **kwargs):
        """Delete internally, connect externally. This baseline has all true class labels
        (train and test) available.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        labels:
            node labels
        n_perturbations : int
            Number of edge removals/additions.

        Returns
        -------
        None.

        """

        # ori_adj: sp.csr_matrix

        print('number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        nonzero = set(zip(*ori_adj.nonzero()))
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]] == labels[x[1]]]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T

            # filter out existing edges, and pairs with the different labels
            candidate_edges = set([(u, v) for u, v in candidate_edges if labels[u] != labels[v]
                                        and modified_adj[u, v] == 0 and modified_adj[v, u] == 0])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj


    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]
