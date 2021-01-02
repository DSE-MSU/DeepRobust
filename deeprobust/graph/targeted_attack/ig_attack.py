"""
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense
        https://arxiv.org/pdf/1903.01610.pdf
"""

import torch
import torch.multiprocessing as mp
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp

class IGAttack(BaseAttack):
    """IGAttack: IG-FGSM. Adversarial Examples on Graph Data: Deep Insights into Attack and Defense, https://arxiv.org/pdf/1903.01610.pdf.

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
    >>> from deeprobust.graph.targeted_attack import IGAttack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = IGAttack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5, steps=10)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features

    """

    def __init__(self, model, nnodes=None, feature_shape=None, attack_structure=True, attack_features=True, device='cpu'):

        super(IGAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None
        self.target_node = None

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train:
            training nodes indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        steps : int
            steps for computing integrated gradients
        """

        self.surrogate.eval()
        self.target_node = target_node


        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        adj, features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)
        adj_norm = utils.normalize_adj_tensor(adj)

        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]
        self.pseudo_labels = pseudo_labels

        s_e = np.zeros(adj.shape[1])
        s_f = np.zeros(features.shape[1])
        if self.attack_structure:
            s_e = self.calc_importance_edge(features, adj_norm, labels, steps)
        if self.attack_features:
            s_f = self.calc_importance_feature(features, adj_norm, labels, steps)

        for t in (range(n_perturbations)):
            s_e_max = np.argmax(s_e)
            s_f_max = np.argmax(s_f)

            if s_e[s_e_max] >= s_f[s_f_max]:
                # edge perturbation score is larger
                if self.attack_structure:
                    value = np.abs(1 - modified_adj[target_node, s_e_max])
                    modified_adj[target_node, s_e_max] = value
                    modified_adj[s_e_max, target_node] = value
                    s_e[s_e_max] = 0
                else:
                    raise Exception("""No posisble perturbation on the structure can be made!
                            See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")
            else:
                # feature perturbation score is larger
                if self.attack_features:
                    modified_features[target_node, s_f_max] = np.abs(1 - modified_features[target_node, s_f_max])
                    s_f[s_f_max] = 0
                else:
                    raise Exception("""No posisble perturbation on the features can be made!
                            See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")


        self.modified_adj = sp.csr_matrix(modified_adj)
        self.modified_features = sp.csr_matrix(modified_features)
        self.check_adj(modified_adj)

    def calc_importance_edge(self, features, adj_norm, labels, steps):
        """Calculate integrated gradient for edges. Although I think the the gradient should be
        with respect to adj instead of adj_norm, but the calculation is too time-consuming. So I
        finally decided to calculate the gradient of loss with respect to adj_norm
        """
        baseline_add = adj_norm.clone()
        baseline_remove = adj_norm.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        adj_norm.requires_grad = True
        integrated_grad_list = []

        i = self.target_node
        for j in tqdm(range(adj_norm.shape[1])):
            if adj_norm[i][j]:
                scaled_inputs = [baseline_remove + (float(k)/ steps) * (adj_norm - baseline_remove) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [baseline_add - (float(k)/ steps) * (baseline_add - adj_norm) for k in range(0, steps + 1)]
            _sum = 0

            for new_adj in scaled_inputs:
                output = self.surrogate(features, new_adj)
                loss = F.nll_loss(output[[self.target_node]],
                        self.pseudo_labels[[self.target_node]])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                adj_grad = adj_grad[i][j]
                _sum += adj_grad

            if adj_norm[i][j]:
                avg_grad = (adj_norm[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - adj_norm[i][j]) * _sum.mean()

            integrated_grad_list.append(avg_grad.detach().item())

        integrated_grad_list[i] = 0
        # make impossible perturbation to be negative
        integrated_grad_list = np.array(integrated_grad_list)
        adj = (adj_norm > 0).cpu().numpy()
        integrated_grad_list = (-2 * adj[self.target_node] + 1) * integrated_grad_list
        integrated_grad_list[self.target_node] = -10
        return integrated_grad_list

    def calc_importance_feature(self, features, adj_norm, labels, steps):
        """Calculate integrated gradient for features
        """
        baseline_add = features.clone()
        baseline_remove = features.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0

        features.requires_grad = True
        integrated_grad_list = []
        i = self.target_node
        for j in tqdm(range(features.shape[1])):
            if features[i][j]:
                scaled_inputs = [baseline_add + (float(k)/ steps) * (features - baseline_add) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [baseline_remove - (float(k)/ steps) * (baseline_remove - features) for k in range(0, steps + 1)]
            _sum = 0

            for new_features in scaled_inputs:
                output = self.surrogate(new_features, adj_norm)
                loss = F.nll_loss(output[[self.target_node]],
                        self.pseudo_labels[[self.target_node]])

                feature_grad = torch.autograd.grad(loss, features)[0]
                feature_grad = feature_grad[i][j]
                _sum += feature_grad

            if features[i][j]:
                avg_grad = (features[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - features[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        # make impossible perturbation to be negative
        features = (features > 0).cpu().numpy()
        integrated_grad_list = np.array(integrated_grad_list)
        integrated_grad_list = (-2 * features[self.target_node] + 1) * integrated_grad_list
        return integrated_grad_list

