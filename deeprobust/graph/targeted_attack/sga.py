import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.targeted_attack import BaseAttack
from deeprobust.graph import utils
from torch_scatter import scatter_add
from collections import namedtuple

from numba import njit
from numba import types
from numba.typed import Dict

SubGraph = namedtuple('SubGraph', ['edge_index', 'non_edge_index',
                                   'self_loop', 'self_loop_weight',
                                   'edge_weight', 'non_edge_weight',
                                   'edges_all'])


class SGAttack(BaseAttack):
    """SGAttack proposed in `Adversarial Attack on Large Scale Graph` TKDE 2021
    <https://arxiv.org/abs/2009.03488>

    SGAttack follows these steps::
    + training a surrogate SGC model with hop K
    + extrack a K-hop subgraph centered at target node
    + choose top-N attacker nodes that belong to the best wrong classes of target node
    + compute gradients w.r.t to the subgraph to add or remove edges

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
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> surrogate = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> surrogate = surrogate.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> surrogate.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    >>> from deeprobust.graph.targeted_attack import SGAttack
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = SGAttack(surrogate, attack_structure=True, device=device)
    >>> # Attack
    >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features
    """

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):

        assert not attack_features, 'Currently `SGAttack` does not support `attack_features`.'
        super(SGAttack, self).__init__(model=None, nnodes=nnodes,
                                       attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.target_node = None
        self.logits = model.predict()
        self.K = model.conv1.K
        W = model.conv1.lin.weight.to(device)
        b = model.conv1.lin.bias
        if b is not None:
            b = b.to(device)

        self.weight, self.bias = W, b

    def get_linearized_weight(self, features):
        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device)
        return F.linear(features, self.weight), self.bias

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=3, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        features :
            Original (unperturbed) node feature matrix
        adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        target_node : int
            target_node node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers : int
            number of the top influencers to choose. For direct attack, it will set as `n_perturbations`.
        """
        if sp.issparse(features):
            # to dense numpy matrix
            features = features.A

        if torch.is_tensor(adj):
            adj = utils.to_scipy(adj).csr()

        modified_adj = adj.copy().tolil()

        target_label = torch.LongTensor([labels[target_node]])
        labels = torch.tensor(labels)
        best_wrong_label = torch.LongTensor([(self.logits[target_node].cpu() - 1000 * torch.eye(self.logits.size(1))[target_label]).argmax()])
        self.selfloop_degree = torch.tensor(adj.sum(1).A1 + 1, device=self.device)
        self.target_label = target_label.to(self.device)
        self.best_wrong_label = best_wrong_label.to(self.device)
        self.n_perturbations = n_perturbations
        self.W, self.b = self.get_linearized_weight(features)
        self.ori_adj = adj
        self.target_node = target_node
        self.direct = direct

        attacker_nodes = torch.where(labels == best_wrong_label)[0]
        subgraph = self.get_subgraph(attacker_nodes, n_influencers)

        if not direct:
            # for indirect attack, the edges adjacent to targeted node should not be considered
            mask = torch.logical_and(subgraph.edge_index[0] != target_node, subgraph.edge_index[1] != target_node).float().to(self.device)
        else:
            mask = 1.0

        structure_perturbations = []
        for _ in range(n_perturbations):
            edge_grad, non_edge_grad = self.compute_gradient(subgraph)
            with torch.no_grad():
                edge_grad *= (-2 * subgraph.edge_weight + 1) * mask
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1

            max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
            max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)

            if max_edge_grad > max_non_edge_grad:
                # remove one edge
                best_edge = subgraph.edge_index[:, max_edge_idx]
                subgraph.edge_weight.data[max_edge_idx] = 0.0
                self.selfloop_degree[best_edge] -= 1.0
            else:
                # add one edge
                best_edge = subgraph.non_edge_index[:, max_non_edge_idx]
                subgraph.non_edge_weight.data[max_non_edge_idx] = 1.0
                self.selfloop_degree[best_edge] += 1.0

            u, v = best_edge.tolist()
            structure_perturbations.append((u, v))
            modified_adj[u, v] = modified_adj[v, u] = 1 - modified_adj[u, v]

        self.modified_adj = modified_adj
        self.modified_features = features
        self.structure_perturbations = structure_perturbations

    def get_subgraph(self, attacker_nodes, n_influencers=None):
        target_node = self.target_node
        neighbors = self.ori_adj[target_node].indices

        sub_edges, sub_nodes = self.ego_subgraph()

        if self.direct or n_influencers is not None:
            influencers = [target_node]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors

        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)

        if n_influencers is not None:
            if self.direct:
                influencers = [target_node]
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations + 1)

            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)

            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph

    def get_topk_influencers(self, subgraph, k):
        _, non_edge_grad = self.compute_gradient(subgraph)
        _, topk_nodes = torch.topk(non_edge_grad, k=k, sorted=False)

        influencers = subgraph.non_edge_index[1][topk_nodes.cpu()]
        return influencers

    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):

        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if len(influencers) > 1:
            mask = self.ori_adj[non_edges[0],
                                non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        nodes = np.union1d(sub_nodes, attacker_nodes)
        self_loop = np.row_stack([nodes, nodes])

        edges_all = np.hstack([sub_edges, sub_edges[[1, 0]], non_edges,
                               non_edges[[1, 0]], self_loop
                               ])

        edges_all = torch.tensor(edges_all, device=self.device)
        edge_weight = nn.Parameter(torch.ones(sub_edges.shape[1], device=self.device))
        non_edge_weight = nn.Parameter(torch.zeros(non_edges.shape[1], device=self.device))
        self_loop_weight = torch.ones(nodes.shape[0], device=self.device)

        edge_index = torch.tensor(sub_edges)
        non_edge_index = torch.tensor(non_edges)
        self_loop = torch.tensor(self_loop)

        subgraph = SubGraph(edge_index=edge_index, non_edge_index=non_edge_index,
                            self_loop=self_loop, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            self_loop_weight=self_loop_weight)
        return subgraph

    def SGCCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x

    def compute_gradient(self, subgraph, eps=5.0):
        edge_weight = subgraph.edge_weight
        non_edge_weight = subgraph.non_edge_weight
        self_loop_weight = subgraph.self_loop_weight
        weights = torch.cat([edge_weight, edge_weight,
                             non_edge_weight, non_edge_weight,
                             self_loop_weight], dim=0)

        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree)
        logit = self.SGCCov(self.W, subgraph.edges_all, weights)
        logit = logit[self.target_node]
        if self.b is not None:
            logit += self.b
        # model calibration
        logit = F.log_softmax(logit.view(1, -1) / eps, dim=1)
        loss = F.nll_loss(logit, self.target_label) - F.nll_loss(logit, self.best_wrong_label)
        gradients = torch.autograd.grad(loss, [edge_weight, non_edge_weight], create_graph=False)
        return gradients

    def ego_subgraph(self):
        import graphgallery.functional as gf
        sub_edges, sub_nodes = gf.ego_graph(self.ori_adj, self.target_node, self.K)
        sub_edges = sub_edges.T  # shape [2, M]
        return sub_edges, sub_nodes

    @staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights


@njit
def extra_edges(indices, indptr,
                last_level, seen,
                hops: int):
    edges = []
    mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for u in last_level:
        nbrs = indices[indptr[u]:indptr[u + 1]]
        nbrs = nbrs[seen[nbrs] == hops]
        mapping[u] = 1
        for v in nbrs:
            if not v in mapping:
                edges.append((u, v))
    return edges


def ego_graph(adj_matrix, targets, hops: int = 1):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    adj_matrix : A Scipy sparse adjacency matrix
        representing a graph

    targets : Center nodes
        A single node or a list of nodes

    hops : number, optional
        Include all neighbors of distance<=hops from nodes.

    Returns
    -------
    (edges, nodes):
        edges: shape [2, M], the edges of the subgraph
        nodes: shape [N], the nodes of the subgraph

    Notes
    -----
    This is a faster implementation of 
    `networkx.ego_graph`


    See Also
    --------
    networkx.ego_graph

    """

    if np.ndim(targets) == 0:
        targets = [targets]
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    else:
        targets = list(targets)

    indices = adj_matrix.indices
    indptr = adj_matrix.indptr

    edges = {}
    start = 0
    N = adj_matrix.shape[0]
    seen = np.zeros(N) - 1
    seen[targets] = 0
    for level in range(hops):
        end = len(targets)
        while start < end:
            head = targets[start]
            nbrs = indices[indptr[head]:indptr[head + 1]]
            for u in nbrs:
                if seen[u] < 0:
                    targets.append(u)
                    seen[u] = level + 1
                if (u, head) not in edges:
                    edges[(head, u)] = level + 1

            start += 1

    if len(targets[start:]):
        e = extra_edges(indices, indptr, np.array(targets[start:]), seen, hops)
    else:
        e = []

    return np.transpose(list(edges.keys()) + e), np.asarray(targets)
