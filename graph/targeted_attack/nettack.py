'''
    Adversarial Attacks on Neural Networks for Graph Data. KDD 2018.
        https://arxiv.org/pdf/1805.07984.pdf
    Author's Implementation
        https://github.com/danielzuegner/nettack

    Since pytorch does not have good enough support to the operations on sparse tensor,
this part of code is heavily based on the author's implementation.
'''

import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from DeepRobust.graph import utils
import torch.nn.functional as F
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from numba import jit
from torch import spmm

class Nettack(BaseAttack):

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='gpu'):

        super(Nettack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

        self.cooc_constraint = None

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(self.nnodes, 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        # t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        t_possible_edges = self.potential_edges
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio

    def get_linearized_weight(self):
        surrogate = self.surrogate

        import ipdb
        ipdb.set_trace()

        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers= 0, ll_cutoff=0.004):

    # def forward(self, features, adj, labels, idx_train, n_perturbations, target_node, direct=True, n_influencers=0, delta_cutoff=0.004, verbose=True):
        """
        Perform an attack on the surrogate model.
        """

        if self.nnodes is None:
            self.nnodes = adj.shape[0]

        self.target_node = target_node
        # ori_adj = adj
        # modified_adj = deepcopy(adj)
        self.ori_adj = utils.to_scipy(adj).tolil()
        self.modified_adj = utils.to_scipy(adj).tolil()
        self.ori_features = utils.to_scipy(features).tolil()
        self.modified_features = utils.to_scipy(features).tolil()
        self.cooc_matrix = self.modified_features.T.dot(self.modified_features).tolil()

        attack_features = self.attack_features
        attack_structure = self.attack_structure
        assert not (direct==False and n_influencers==0), "indirect mode requires at least one influencer node"
        assert n_perturbations > 0, "need at least one perturbation"
        assert attack_features or attack_structure, "either attack_features or attack_structure must be true"

        # adj_norm = utils.normalize_adj_tensor(modified_adj, sparse=True)
        self.adj_norm = utils.normalize_adj(self.modified_adj)

        self.W = self.get_linearized_weight()

        logits = (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W )[target_node]

        self.label_u = labels[target_node]
        label_target_onehot = np.eye(int(self.nclass))[labels[target_node]]
        best_wrong_class = (logits - 1000*label_target_onehot).argmax()
        surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]

        verbose=True
        if verbose:
            print("##### Starting attack #####")
            if attack_structure and attack_features:
                print("##### Attack node with ID {} using structure and feature perturbations #####".format(target_node))
            elif attack_features:
                print("##### Attack only using feature perturbations #####")
            elif attack_structure:
                print("##### Attack only using structure perturbations #####")
            if direct:
                print("##### Attacking the node directly #####")
            else:
                print("##### Attacking the node indirectly via {} influencer nodes #####".format(n_influencers))
            print("##### Performing {} perturbations #####".format(n_perturbations))

        # ori_adj_scipy = to_scipy(ori_adj)
        # modified_adj_scipy = to_scipy(modified_adj)
        if attack_structure:
            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.ori_adj.sum(0).A1
            current_degree_sequence = self.modified_adj.sum(0).A1
            d_min = 2

            # ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)

            S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)

            log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        if len(self.influencer_nodes) == 0:
            if not direct:
                # Choose influencer nodes
                infls, add_infls = self.get_attacker_nodes(n_influencers, add_additional_nodes=True)
                self.influencer_nodes = np.concatenate((infls, add_infls)).astype("int")
                # Potential edges are all edges from any attacker to any other node, except the respective
                # attacker itself or the node being attacked.
                self.potential_edges = np.row_stack([np.column_stack((np.tile(infl, self.nnodes - 2),
                                                            np.setdiff1d(np.arange(self.nnodes),
                                                            np.array([target_node,infl])))) for infl in
                                                            self.influencer_nodes])
                if verbose:
                    print("Influencer nodes: {}".format(self.influencer_nodes))
            else:
                # direct attack
                influencers = [target_node]
                self.potential_edges = np.column_stack((np.tile(target_node, self.nnodes-1), np.setdiff1d(np.arange(self.nnodes), target_node)))
                self.influencer_nodes = np.array(influencers)

        self.potential_edges = self.potential_edges.astype("int32")

        for _ in range(n_perturbations):
            if verbose:
                print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
            if attack_structure:

                # Do not consider edges that, if removed, result in singleton edges in the graph.
                singleton_filter = filter_singletons(self.potential_edges, self.modified_adj)
                filtered_edges = self.potential_edges[singleton_filter]

                # Update the values for the power law likelihood ratio test.

                deltas = 2 * (1 - self.modified_adj[tuple(filtered_edges.T)].toarray()[0] )- 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
                new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
                alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi_square cutoff value.
                powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]

                # allowed_mask, ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, delta_cutoff)
                # allowed_mask = allowed_mask.to(self.device)
                # filtered_edges_final = allowed_mask * self.filter_potential_singletons(modified_adj)

                # filtered_edges_final = self.potential_edges

                # Compute new entries in A_hat_square_uv
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final, target_node)
                # Compute the struct scores for each potential edge
                struct_scores = self.struct_score(a_hat_uv_new, self.modified_features @ self.W)
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges_final[best_edge_ix]

            if attack_features:
                # Compute the feature scores for each potential feature perturbation
                feature_ixs, feature_scores = self.feature_scores()
                best_feature_ix = feature_ixs[0]
                best_feature_score = feature_scores[0]

            if attack_structure and attack_features:
                # decide whether to choose an edge or feature to change
                if best_edge_score < best_feature_score:
                    if verbose:
                        print("Edge perturbation: {}".format(best_edge))
                    change_structure = True
                else:
                    if verbose:
                        print("Feature perturbation: {}".format(best_feature_ix))
                    change_structure=False

            elif attack_structure:
                change_structure = True
            elif attack_features:
                change_structure = False

            if change_structure:
                # perform edge perturbation
                self.modified_adj[tuple(best_edge)] = self.modified_adj[tuple(best_edge[::-1])] = 1 - self.modified_adj[tuple(best_edge)]
                self.adj_norm = utils.normalize_adj(self.modified_adj)

                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                surrogate_losses.append(best_edge_score)

                # Update likelihood ratio test values
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]

            else:
                self.modified_features[tuple(best_feature_ix)] = 1 - self.modified_features[tuple(best_feature_ix)]
                self.feature_perturbations.append(tuple(best_feature_ix))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)

        return self.modified_adj


    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[self.target_node]

    def strongest_wrong_class(self, logits):
        label_u_onehot = np.eye(self.nclass)[self.label_u]
        return (logits - 1000*label_u_onehot).argmax()

    def feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """

        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]

        gradient = self.gradient_wrt_x(self.label_u) - self.gradient_wrt_x(best_wrong_class)
        # gradients_flipped = (gradient * -1).tolil()
        gradients_flipped = sp.lil_matrix(gradient * -1)
        gradients_flipped[self.modified_features.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.modified_features.shape)
        X_influencers[self.influencer_nodes] = self.modified_features[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply((self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def compute_cooccurrence_constraint(self, nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array [len(nodes), D], dtype bool
            Binary matrix of dimension len(nodes) x D. A 1 in entry n,d indicates that
            we are allowed to add feature d to the features of node n.

        """

        words_graph = self.cooc_matrix.copy()
        D = self.modified_features.shape[1]
        words_graph.setdiag(0)
        words_graph = (words_graph > 0)
        word_degrees = np.sum(words_graph, axis=0).A1

        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)

        sd = np.zeros([self.nnodes])
        for n in range(self.nnodes):
            n_idx = self.modified_features[n, :].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])

        scores_matrix = sp.lil_matrix((self.nnodes, D))

        for n in nodes:
            common_words = words_graph.multiply(self.modified_features[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array([idegs[nnz == ix].sum() for ix in range(D)])
            scores_matrix[n] = scores
        self.cooc_constraint = sp.csr_matrix(scores_matrix - 0.5 * sd[:, None] > 0)

    def gradient_wrt_x(self, label):
        # return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].T)
        return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].reshape(1, -1))

    def reset(self):
        """
        Reset Nettack
        """
        self.modified_adj = self.ori_adj.copy()
        self.modified_features = self.ori_features.copy()
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def compute_new_a_hat_uv(self, potential_edges, adj):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """
        edges = np.array(adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees, potential_edges, target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])

        return a_hat_uv

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def compute_new_a_hat_uv(self, potential_edges, target_node):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(self.modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges, target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])

        return a_hat_uv

@jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@jit(nopython=True)
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    node_nb_ixs: np.array, shape [N,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    values_before: np.array, shape [N,], the values in [A_hat]^2_uv to be updated.
    degs: np.array, shape [N,], dtype int
        The degree of the nodes in the input graph.
    potential_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    u: int
        The target node

    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, N] matrix of updated values that have changed
    return_values:

    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.

    Parameters
    ----------
    edges: np.array, shape [P, 2], dtype int, where P is the number of input edges.
        The potential edges.

    adj: sp.sparse_matrix, shape [N,N]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [P, 2], dtype bool:
        A binary vector of length len(edges), False values indicate that the edge at
        the index  generates singleton edges, and should thus be avoided.

    """


    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution.

    Parameters
    ----------
    n: int or np.array of int
        Number of entries that are larger than or equal to d_min

    S_d: float or np.array of float
         Sum of log degrees in the distribution that are larger than or equal to d_min

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.

    Parameters
    ----------
    S_old: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    n_old: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    d_old: np.array, shape [N,] dtype int
        The old degree sequence.

    d_new: np.array, shape [N,] dtype int
        The new degree sequence

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
    new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
    """

    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit.

    Parameters
    ----------
    n: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    alpha: float
        The estimated alpha of the power law distribution

    S_d: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    float: the estimated log likelihood
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff
