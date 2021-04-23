import numpy as np
import torch
from .dataset import Dataset
import scipy.sparse as sp
from itertools import repeat
import os.path as osp
import warnings
import sys
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Coauthor, Amazon

class Dpr2Pyg(InMemoryDataset):
    """Convert deeprobust data (sparse matrix) to pytorch geometric data (tensor, edge_index)

    Parameters
    ----------
    dpr_data :
        data instance of class from deeprobust.graph.data, e.g., deeprobust.graph.data.Dataset,
        deeprobust.graph.data.PtbDataset, deeprobust.graph.data.PrePtbDataset
    transform :
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before every access. For example, you can
        use torch_geometric.transforms.NormalizeFeatures()

    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format.

    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    """

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = index_to_mask(idx_train, size=y.size(0))
        val_mask = index_to_mask(idx_val, size=y.size(0))
        test_mask = index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def update_edge_index(self, adj):
        """ This is an inplace operation to substitute the original edge_index
        with adj.nonzero()

        Parameters
        ----------
        adj: sp.csr_matrix
            update the original adjacency into adj (by change edge_index)
        """
        self.data.edge_index = torch.LongTensor(adj.nonzero())
        self.data, self.slices = self.collate([self.data])

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

class Pyg2Dpr(Dataset):
    """Convert pytorch geometric data (tensor, edge_index) to deeprobust
    data (sparse matrix)

    Parameters
    ----------
    pyg_data :
        data instance of class from pytorch geometric dataset

    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format and then convert it back to Dataset class.

    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    >>> dpr_data = Pyg2Dpr(pyg_data)
    >>> print(dpr_data.adj)
    """

    def __init__(self, pyg_data, **kwargs):
        is_ogb = hasattr(pyg_data, 'get_idx_split')
        if is_ogb: # get splits for ogb datasets
            splits = pyg_data.get_idx_split()
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()
        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
        if is_ogb: # set splits for ogb datasets
            self.idx_train = splits['train'].numpy()
            self.idx_val = splits['valid'].numpy()
            self.idx_test = splits['test'].numpy()
        else:
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
        self.name = 'Pyg2Dpr'

class AmazonPyg(Amazon):
    """Amazon-Computers and Amazon-Photo datasets loaded from pytorch geomtric;
    the way we split the dataset follows Towards Deeper Graph Neural Networks
    (https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/train_eval.py).
    Specifically, 20 * num_classes labels for training, 30 * num_classes labels
    for validation, rest labels for testing.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be choosen from ['computers', 'photo']
    transform :
        A function/transform that takes in an torch_geometric.data.Data object
        and returns a transformed version. The data object will be transformed
        before every access. (default: None)
    pre_transform :
         A function/transform that takes in an torch_geometric.data.Data object
         and returns a transformed version. The data object will be transformed
         before being saved to disk.

    Examples
    --------
    We can directly load Amazon dataset from deeprobust in the format of pyg.

    >>> from deeprobust.graph.data import AmazonPyg
    >>> computers = AmazonPyg(root='/tmp', name='computers')
    >>> print(computers)
    >>> print(computers[0])
    >>> photo = AmazonPyg(root='/tmp', name='photo')
    >>> print(photo)
    >>> print(photo[0])
    """

    def __init__(self, root, name, transform=None, pre_transform=None, **kwargs):
        path = osp.join(root, 'pygdata', name)
        super(AmazonPyg, self).__init__(path, name, transform, pre_transform)

        random_coauthor_amazon_splits(self, self.num_classes, lcc_mask=None)
        self.data, self.slices = self.collate([self.data])


class CoauthorPyg(Coauthor):
    """Coauthor-CS and Coauthor-Physics datasets loaded from pytorch geomtric;
    the way we split the dataset follows Towards Deeper Graph Neural Networks
    (https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/train_eval.py).
    Specifically, 20 * num_classes labels for training, 30 * num_classes labels
    for validation, rest labels for testing.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be choosen from ['cs', 'physics']
    transform :
        A function/transform that takes in an torch_geometric.data.Data object
        and returns a transformed version. The data object will be transformed
        before every access. (default: None)
    pre_transform :
         A function/transform that takes in an torch_geometric.data.Data object
         and returns a transformed version. The data object will be transformed
         before being saved to disk.

    Examples
    --------
    We can directly load Coauthor dataset from deeprobust in the format of pyg.

    >>> from deeprobust.graph.data import CoauthorPyg
    >>> cs = CoauthorPyg(root='/tmp', name='cs')
    >>> print(cs)
    >>> print(cs[0])
    >>> physics = CoauthorPyg(root='/tmp', name='physics')
    >>> print(physics)
    >>> print(physics[0])
    """

    def __init__(self, root, name, transform=None, pre_transform=None, **kwargs):
        path = osp.join(root, 'pygdata', name)
        super(CoauthorPyg, self).__init__(path, name, transform, pre_transform)
        random_coauthor_amazon_splits(self, self.num_classes, lcc_mask=None)
        self.data, self.slices = self.collate([self.data])


def random_coauthor_amazon_splits(dataset, num_classes, lcc_mask):
    """https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/train_eval.py
    Set random coauthor/co-purchase splits:
    * 20 * num_classes labels for training
    * 30 * num_classes labels for validation
    rest labels for testing
    """
    data = dataset.data
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


if __name__ == "__main__":
    from deeprobust.graph.data import PrePtbDataset, Dataset
    # load clean graph data
    dataset_str = 'cora'
    data = Dataset(root='/tmp/', name=dataset_str, seed=15)
    pyg_data = Dpr2Pyg(data)
    print(pyg_data)
    print(pyg_data[0])
    dpr_data = Pyg2Dpr(pyg_data)
    print(dpr_data)

    computers = AmazonPyg(root='/tmp', name='computers')
    print(computers)
    print(computers[0])
    photo = AmazonPyg(root='/tmp', name='photo')
    print(photo)
    print(photo[0])
    cs = CoauthorPyg(root='/tmp', name='cs')
    print(cs)
    print(cs[0])
    physics = CoauthorPyg(root='/tmp', name='physics')
    print(physics)
    print(physics[0])

    # from ogb.nodeproppred import PygNodePropPredDataset
    # dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
    # ogb_data = Pyg2Dpr(dataset)


