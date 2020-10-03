"""
    FGA: Fast Gradient Attack on Network Embedding (https://arxiv.org/pdf/1809.02797.pdf)
    Another very similar algorithm to mention here is FGSM (for graph data).
    It is mentioned in Zugner's paper,
    Adversarial Attacks on Neural Networks for Graph Data, KDD'19
"""

import torch
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from deeprobust.graph import utils
import torch.nn.functional as F
import scipy.sparse as sp

class FGA(BaseAttack):
    """FGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
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
    >>> from deeprobust.graph.targeted_attack import FGA
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(FGA, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)


        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, verbose=False, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        labels :
            node labels
        idx_train:
            training node indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """

        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        modified_adj, modified_features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)

        self.surrogate.eval()
        if verbose == True:
            print('number of pertubations: %s' % n_perturbations)

        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]

        modified_adj.requires_grad = True
        for i in range(n_perturbations):
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            if self.attack_structure:
                output = self.surrogate(modified_features, adj_norm)
                loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
                grad = torch.autograd.grad(loss, modified_adj)[0]
                # bidirection
                grad = (grad[target_node] + grad[:, target_node]) * (-2*modified_adj[target_node] + 1)
                grad[target_node] = -10
                grad_argmax = torch.argmax(grad)

            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value

            if self.attack_features:
                pass

        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        # self.modified_features = modified_features


