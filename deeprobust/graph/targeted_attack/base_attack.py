from torch.nn.modules.module import Module
import numpy as np
import torch
import scipy.sparse as sp
import os.path as osp

class BaseAttack(Module):
    """Abstract base class for target attack classes.

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

    """

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        self.modified_adj = None
        self.modified_features = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.

        Returns
        -------
        None.

        """
        pass

    def check_adj(self, adj):
        """Check if the modified adjacency is symmetric and unweighted.
        """

        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        if sp.issparse(adj):
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"
        else:
            assert adj.max() == 1, "Max value should be 1!"
            assert adj.min() == 0, "Min value should be 0!"

    def save_adj(self, root=r'/tmp/', name='mod_adj'):
        """Save attacked adjacency matrix.

        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name

        Returns
        -------
        None.

        """
        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            modified_adj = utils.to_scipy(modified_adj)
        if sp.issparse(modified_adj):
            modified_adj = modified_adj.tocsr()
        sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):
        """Save attacked node feature matrix.

        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name

        Returns
        -------
        None.

        """

        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            modified_features = utils.to_scipy(modified_features)
        if sp.issparse(modified_features):
            modified_features = modified_features.tocsr()
        sp.save_npz(osp.join(root, name), modified_features)

