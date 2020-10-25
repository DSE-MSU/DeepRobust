import numpy as np
from deeprobust.graph.global_attack import BaseAttack
import scipy.sparse as sp
# import random


class Random(BaseAttack):
    """ Randomly adding edges to the input graph

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
    >>> from deeprobust.graph.global_attack import Random
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = Random()
    >>> model.attack(adj, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, ori_adj, n_perturbations, type='add', **kwargs):
        """Generate attacks on the input graph.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        -------
        None.

        """

        if self.attack_structure:
            modified_adj = self.perturb_adj(ori_adj, n_perturbations, type)
            self.modified_adj = modified_adj

    def perturb_adj(self, adj, n_perturbations, type='add'):
        """Randomly add, remove or flip edges.

        Parameters
        ----------
        adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of edge removals/additions.
        type: str
            perturbation type. Could be 'add', 'remove' or 'flip'.

        Returns
        ------
        scipy.sparse matrix
            perturbed adjacency matrix
        """
        # adj: sp.csr_matrix
        modified_adj = adj.tolil()

        type = type.lower()
        assert type in ['add', 'remove', 'flip']

        if type == 'flip':
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

        if type == 'add':
            # sample edges to add
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1

        if type == 'remove':
            # sample edges to remove
            nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
            indices = np.random.permutation(nonzero)[: n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0

        self.check_adj(modified_adj)
        return modified_adj

    def perturb_features(self, features, n_perturbations):
        """Randomly perturb features.
        """
        raise NotImplementedError
        print('number of pertubations: %s' % n_perturbations)
        return modified_features

    def inject_nodes(self, adj, n_add, n_perturbations):
        """For each added node, randomly connect with other nodes.
        """
        # adj: sp.csr_matrix
        # TODO
        print('number of pertubations: %s' % n_perturbations)
        raise NotImplementedError

        modified_adj = adj.tolil()
        return modified_adj

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))
