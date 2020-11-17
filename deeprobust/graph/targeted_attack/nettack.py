"""
    Adversarial Attacks on Neural Networks for Graph Data. KDD 2018.
        https://arxiv.org/pdf/1805.07984.pdf
    Author's Implementation
        https://github.com/danielzuegner/nettack

    Since pytorch does not have good enough support to the operations
    on sparse tensor, this part of code is heavily based on the author's implementation.
"""
"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Neural Networks for Graph Data'
by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann,
published at SIGKDD'18, August 2018, London, UK
Copyright (C) 2018
Daniel Zügner
Technical University of Munich
"""

import torch
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
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
    """Nettack.

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
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.targeted_attack import Nettack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features

    """

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):

        super(Nettack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

        self.cooc_constraint = None

    def filter_potential_singletons(self, modified_adj):
        """Computes a mask for entries potentially leading to singleton nodes, i.e.
        one of the two nodes corresponding to the entry have degree 1 and there
        is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(self.nnodes, 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers= 0, ll_cutoff=0.004, verbose=True, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : torch.Tensor or scipy.sparse.csr_matrix
            Origina (unperturbed) node feature matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        ori_adj : torch.Tensor or scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        labels :
            node labels
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers:
            number of influencer nodes when performing indirect attack.
            (setting `direct` to False). When `direct` is True, it would be ignored.
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95.
        verbose : bool
            whether to show verbose logs
        """

        if self.nnodes is None:
            self.nnodes = adj.shape[0]

        self.target_node = target_node

        if type(adj) is torch.Tensor:
            self.ori_adj = utils.to_scipy(adj).tolil()
            self.modified_adj = utils.to_scipy(adj).tolil()
            self.ori_features = utils.to_scipy(features).tolil()
            self.modified_features = utils.to_scipy(features).tolil()
        else:
            self.ori_adj = adj.tolil()
            self.modified_adj = adj.tolil()
            self.ori_features = features.tolil()
            self.modified_features = features.tolil()

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

        if attack_structure:
            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.ori_adj.sum(0).A1
            current_degree_sequence = self.modified_adj.sum(0).A1
            d_min = 2

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

        # return self.modified_adj, self.modified_features

    def get_attacker_nodes(self, n=5, add_additional_nodes = False):
        """Determine the influencer nodes to attack node i based on
        the weights W and the attributes X.
        """
        assert n < self.nnodes-1, "number of influencers cannot be >= number of nodes in the graph!"
        neighbors = self.ori_adj[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors

        potential_edges = np.column_stack((np.tile(self.target_node, len(neighbors)),neighbors)).astype("int32")

        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors, respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges, self.target_node)

        # XW = self.compute_XW()
        XW = self.modified_features @ self.W

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:

            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.nnodes),neighbors), self.target_node)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n-len(neighbors)
                possible_edges = np.column_stack((np.tile(self.target_node, n_possible_additional), poss_add_infl))

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges, self.target_node)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[self.target_node]

    def strongest_wrong_class(self, logits):
        label_u_onehot = np.eye(self.nclass)[self.label_u]
        return (logits - 1000*label_u_onehot).argmax()

    def feature_scores(self):
        """Compute feature scores for all possible feature changes.
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
        """Reset Nettack
        """
        self.modified_adj = self.ori_adj.copy()
        self.modified_features = self.ori_features.copy()
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None


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
                                         potential_edges.astype(np.int32), target_node)
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

    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.
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

    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff
