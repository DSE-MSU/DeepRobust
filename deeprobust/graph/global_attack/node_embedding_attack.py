"""
Code in this file is modified from https://github.com/abojchevski/node_embedding_attack

'Adversarial Attacks on Node Embeddings via Graph Poisoning'
Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
http://proceedings.mlr.press/v97/bojchevski19a.html
Copyright (C) owned by the authors, 2019
"""

import numba
import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
import torch
import networkx as nx
from deeprobust.graph.global_attack import BaseAttack


class NodeEmbeddingAttack(BaseAttack):
    """Node embedding attack. Adversarial Attacks on Node Embeddings via Graph
    Poisoning. Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
    http://proceedings.mlr.press/v97/bojchevski19a.html

    Examples
    -----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import NodeEmbeddingAttack
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = NodeEmbeddingAttack()
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="remove", min_span_tree=True)
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="add", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    >>> model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    """

    def __init__(self):
        pass

    def attack(self, adj, n_perturbations=1000, dim=32, window_size=5,
            attack_type="remove", min_span_tree=False, n_candidates=None, seed=None, **kwargs):
        """Selects the top (n_perturbations) number of flips using our perturbation attack.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param n_perturbations: int
            Number of flips to select
        :param dim: int
            Dimensionality of the embeddings.
        :param window_size: int
            Co-occurence window size.
        :param attack_type: str
            can be chosed from ["remove", "add", "add_by_remove"]
        :param min_span_tree: bool
            Whether to disallow edges that lie on the minimum spanning tree;
            only valid when `attack_type` is "remove"
        :param n_candidates: int
            Number of candiates for addition; only valid when `attack_type` is "add" or "add_by_remove";
        :param seed: int
            Random seed
        """
        assert attack_type in ["remove", "add", "add_by_remove"],  \
                "attack_type can only be `remove` or `add`"

        if attack_type == "remove":
            if min_span_tree:
                candidates = self.generate_candidates_removal_minimum_spanning_tree(adj)
            else:
                candidates = self.generate_candidates_removal(adj, seed)

        elif attack_type == "add" or attack_type == "add_by_remove":

            assert n_candidates, "please specify the value of `n_candidates`, " \
                    +  "i.e. how many candiate you want to genereate for addition"
            candidates = self.generate_candidates_addition(adj, n_candidates, seed)


        n_nodes = adj.shape[0]

        if attack_type == "add_by_remove":
            candidates_add = candidates
            adj_add = self.flip_candidates(adj, candidates_add)
            vals_org_add, vecs_org_add = spl.eigh(adj_add.toarray(), np.diag(adj_add.sum(1).A1))
            flip_indicator = 1 - 2 * adj_add[candidates[:, 0], candidates[:, 1]].A1

            loss_est = estimate_loss_with_delta_eigenvals(candidates_add, flip_indicator,
                                                          vals_org_add, vecs_org_add, n_nodes, dim, window_size)

            loss_argsort = loss_est.argsort()
            top_flips = candidates_add[loss_argsort[:n_perturbations]]

        else:
            # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
            delta_w = 1 - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

            # generalized eigenvalues/eigenvectors
            deg_matrix = np.diag(adj.sum(1).A1)
            vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix)

            loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim, window_size)
            top_flips = candidates[loss_for_candidates.argsort()[-n_perturbations:]]

        assert len(top_flips) == n_perturbations

        modified_adj = self.flip_candidates(adj, top_flips)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def generate_candidates_removal(self, adj, seed=None):
        """Generates candidate edge flips for removal (edge -> non-edge),
        disallowing one random edge per node to prevent singleton nodes.

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        n_nodes = adj.shape[0]
        if seed is not None:
            np.random.seed(seed)
        deg = np.where(adj.sum(1).A1 == 1)[0]
        hiddeen = np.column_stack(
            (np.arange(n_nodes), np.fromiter(map(np.random.choice, adj.tolil().rows), dtype=np.int32)))

        adj_hidden = edges_to_sparse(hiddeen, adj.shape[0])
        adj_hidden = adj_hidden.maximum(adj_hidden.T)

        adj_keep = adj - adj_hidden

        candidates = np.column_stack((sp.triu(adj_keep).nonzero()))

        candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

        return candidates

    def generate_candidates_removal_minimum_spanning_tree(self, adj):
        """Generates candidate edge flips for removal (edge -> non-edge),
         disallowing edges that lie on the minimum spanning tree.
        adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        mst = sp.csgraph.minimum_spanning_tree(adj)
        mst = mst.maximum(mst.T)
        adj_sample = adj - mst
        candidates = np.column_stack(sp.triu(adj_sample, 1).nonzero())

        return candidates

    def generate_candidates_addition(self, adj, n_candidates, seed=None):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param n_candidates: int
            Number of candidates to generate.
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        if seed is not None:
            np.random.seed(seed)

        num_nodes = adj.shape[0]

        candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:, 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:n_candidates]

        assert len(candidates) == n_candidates

        return candidates

    def flip_candidates(self, adj, candidates):
        """Flip the edges in the candidate set to non-edges and vise-versa.

        :param adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :return: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph with the flipped edges/non-edges.
        """
        adj_flipped = adj.copy().tolil()
        adj_flipped[candidates[:, 0], candidates[:, 1]] = 1 - adj[candidates[:, 0], candidates[:, 1]]
        adj_flipped[candidates[:, 1], candidates[:, 0]] = 1 - adj[candidates[:, 1], candidates[:, 0]]
        adj_flipped = adj_flipped.tocsr()
        adj_flipped.eliminate_zeros()

        return adj_flipped


@numba.jit(nopython=True)
def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param n_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """

    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est


@numba.jit(nopython=True)
def estimate_delta_eigenvecs(candidates, flip_indicator, degrees, vals_org, vecs_org, delta_eigvals, pinvs):
    """Computes the estimated change in the eigenvectors for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param degrees: np.ndarray, shape [n]
        Vector of node degrees.
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param delta_eigvals: np.ndarray, shape [?, n]
        Estimated change in the eigenvalues for all candidate edge flips
    :param pinvs: np.ndarray, shape [k, n, n]
        Precomputed pseudo-inverse matrices for every dimension
    :return: np.ndarray, shape [?, n, k]
        Estimated change in the eigenvectors for all candidate edge flips
    """
    n_nodes, dim = vecs_org.shape
    n_candidates = len(candidates)
    delta_eigvecs = np.zeros((n_candidates, dim, n_nodes))

    for k in range(dim):
        cur_eigvecs = vecs_org[:, k]
        cur_eigvals = vals_org[k]
        for c in range(n_candidates):
            degree_eigvec = (-delta_eigvals[c, k] * degrees) * cur_eigvecs
            i, j = candidates[c]

            degree_eigvec[i] += cur_eigvecs[j] - cur_eigvals * cur_eigvecs[i]
            degree_eigvec[j] += cur_eigvecs[i] - cur_eigvals * cur_eigvecs[j]

            delta_eigvecs[c, k] = np.dot(pinvs[k], flip_indicator[c] * degree_eigvec)

    return delta_eigvecs


def estimate_delta_eigvals(candidates, adj, vals_org, vecs_org):
    """Computes the estimated change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param adj: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :return: np.ndarray, shape [?, n]
        Estimated change in the eigenvalues for all candidate edge flips
    """
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

    delta_eigvals = delta_w[:, None] * (2 * vecs_org[candidates[:, 0]] * vecs_org[candidates[:, 1]]
                                        - vals_org * (
                                                vecs_org[candidates[:, 0]] ** 2 + vecs_org[candidates[:, 1]] ** 2))

    return delta_eigvals


class OtherNodeEmbeddingAttack(NodeEmbeddingAttack):
    """ Baseline methods from the paper Adversarial Attacks on Node Embeddings
    via Graph Poisoning. Aleksandar Bojchevski and Stephan Günnemann, ICML 2019.
    http://proceedings.mlr.press/v97/bojchevski19a.html

    Examples
    -----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import OtherNodeEmbeddingAttack
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> model = OtherNodeEmbeddingAttack(type='degree')
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> #
    >>> model = OtherNodeEmbeddingAttack(type='eigencentrality')
    >>> model.attack(adj, attack_type="remove")
    >>> modified_adj = model.modified_adj
    >>> #
    >>> model = OtherNodeEmbeddingAttack(type='random')
    >>> model.attack(adj, attack_type="add", n_candidates=10000)
    >>> modified_adj = model.modified_adj
    """

    def __init__(self, type):
        assert type in ["degree", "eigencentrality", "random"]
        self.type = type

    def attack(self, adj, n_perturbations=1000, attack_type="remove",
            min_span_tree=False, n_candidates=None, seed=None, **kwargs):
        """Selects the top (n_perturbations) number of flips using our perturbation attack.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param n_perturbations: int
            Number of flips to select
        :param dim: int
            Dimensionality of the embeddings.
        :param attack_type: str
            can be chosed from ["remove", "add"]
        :param min_span_tree: bool
            Whether to disallow edges that lie on the minimum spanning tree;
            only valid when `attack_type` is "remove"
        :param n_candidates: int
            Number of candiates for addition; only valid when `attack_type` is "add";
        :param seed: int
            Random seed;
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        assert attack_type in ["remove", "add"],  \
                "attack_type can only be `remove` or `add`"

        if attack_type == "remove":
            if min_span_tree:
                candidates = self.generate_candidates_removal_minimum_spanning_tree(adj)
            else:
                candidates = self.generate_candidates_removal(adj, seed)
        elif attack_type == "add":
            assert n_candidates, "please specify the value of `n_candidates`, " \
                    +  "i.e. how many candiate you want to genereate for addition"
            candidates = self.generate_candidates_addition(adj, n_candidates, seed)
        else:
            raise NotImplementedError

        if self.type == "random":
            top_flips = self.random_top_flips(candidates, n_perturbations, seed)
        elif self.type == "eigencentrality":
            top_flips = self.eigencentrality_top_flips(adj, candidates, n_perturbations)
        elif self.type == "degree":
            top_flips = self.degree_top_flips(adj, candidates, n_perturbations, complement=False)
        else:
            raise NotImplementedError

        assert len(top_flips) == n_perturbations
        modified_adj = self.flip_candidates(adj, top_flips)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

    def random_top_flips(self, candidates, n_perturbations, seed=None):
        """Selects (n_perturbations) number of flips at random.

        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :param seed: int
            Random seed
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        if seed is not None:
            np.random.seed(seed)
        return candidates[np.random.permutation(len(candidates))[:n_perturbations]]


    def eigencentrality_top_flips(self, adj, candidates, n_perturbations):
        """Selects the top (n_perturbations) number of flips using eigencentrality score of the edges.
        Applicable only when removing edges.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        edges = np.column_stack(sp.triu(adj, 1).nonzero())
        line_graph = construct_line_graph(adj)
        eigcentrality_scores = nx.eigenvector_centrality_numpy(nx.Graph(line_graph))
        eigcentrality_scores = {tuple(edges[k]): eigcentrality_scores[k] for k, v in eigcentrality_scores.items()}
        eigcentrality_scores = np.array([eigcentrality_scores[tuple(cnd)] for cnd in candidates])
        scores_argsrt = eigcentrality_scores.argsort()
        return candidates[scores_argsrt[-n_perturbations:]]


    def degree_top_flips(self, adj, candidates, n_perturbations, complement):
        """Selects the top (n_perturbations) number of flips using degree centrality score of the edges.

        :param adj: sp.spmatrix
            The graph represented as a sparse scipy matrix
        :param candidates: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        :param n_perturbations: int
            Number of flips to select
        :param complement: bool
            Whether to look at the complement graph
        :return: np.ndarray, shape [?, 2]
            The top edge flips from the candidate set
        """
        if complement:
            adj = sp.csr_matrix(1-adj.toarray())
        deg = adj.sum(1).A1
        deg_argsort = (deg[candidates[:, 0]] + deg[candidates[:, 1]]).argsort()

        return candidates[deg_argsort[-n_perturbations:]]


@numba.jit(nopython=True)
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))

    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)


def edges_to_sparse(edges, num_nodes, weights=None):
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()

def construct_line_graph(adj):
    """Construct a line graph from an undirected original graph.

    Parameters
    ----------
    adj : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.
    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjacency matrix of the line graph.
    """
    N = adj.shape[0]
    edges = np.column_stack(sp.triu(adj, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]

    I = sp.eye(N).tocsr()
    E1 = I[e1]
    E2 = I[e2]

    L = E1.dot(E1.T) + E1.dot(E2.T) + E2.dot(E1.T) + E2.dot(E2.T)

    return L - 2 * sp.eye(L.shape[0])


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import DeepWalk
    import itertools
    # load clean graph data
    dataset_str = 'cora_ml'
    data = Dataset(root='/tmp/', name=dataset_str, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    comb = itertools.product(["random", "degree", "eigencentrality"], ["remove", "add"])
    for type, attack_type in comb:
        model = OtherNodeEmbeddingAttack(type=type)
        print(model.type, attack_type)
        try:
            model.attack(adj, attack_type=attack_type, n_candidates=10000)
            defender = DeepWalk()
            defender.fit(adj)
            defender.evaluate_node_classification(labels, idx_train, idx_test)
        except KeyError:
            print('eigencentrality only supports removing edges')

    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type="remove")
    model.attack(adj, attack_type="remove", min_span_tree=True)
    modified_adj = model.modified_adj
    model.attack(adj, attack_type="add", n_candidates=10000)
    model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    # model.attack(adj, attack_type="add")
