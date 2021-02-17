"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack


class PGDAttack(BaseAttack):
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
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
    >>> from deeprobust.graph.global_attack import PGDAttack
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, epochs=200, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        victim_model.eval()
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)


    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = F.nll_loss(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):

        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


class MinMax(PGDAttack):
    """MinMax attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
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
    >>> from deeprobust.graph.global_attack import MinMax
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features, device=device)


    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        # optimizer
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        epochs = 200
        victim_model.eval()
        for t in tqdm(range(epochs)):
            # update victim model
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate pgd attack
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            # adj_grad = self.adj_changes.grad

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            # self.adj_changes.grad.zero_()
            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
