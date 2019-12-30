import numpy as np
import scipy.sparse as sp
import os.path as osp
import os

class Dataset():

    def __init__(self, root, name, require_lcc=True, transform=None):
        self.name = name.lower()
        assert self.name in ['cora', 'citeseer', 'pubmed', 'polblogs'], \
            'Currently only support cora, citeseer, pubmed, polblogs'

        self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_filename = osp.join(root, self.name)
        self.require_lcc = require_lcc
        self.transform = transform
        self.adj, self.features, self.labels = self.load_data(self.root)

    def load_data(self, val_size=0.1, test_size=0.1):
        self.data_filename += '.npz'
        if not osp.exists(self.data_filename):
            self.download_npz()
        print('Loading {} dataset...'.format(self.name))
        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_npz(self):
        print(f'Dowloading from {self.url} to {self.data_filename}')
        if os.system(f'wget -O {self.data_filename} {self.url}'):
            os.system(f'rm {self.data_filename}')
            raise Exception("Download failed!")

    def get_adj(self, require_lcc=True):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj[adj > 1] = 1
        if require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32")
        adj.eliminate_zeros()

        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        if features is None:
            features = np.eye(adj.shape[0])

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{}()'.format(self.name)


