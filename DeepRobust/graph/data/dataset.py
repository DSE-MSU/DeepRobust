import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request

from DeepRobust.graph.utils import get_train_val_test, get_train_val_test_gcn

class Dataset():

    def __init__(self, root, name, setting='nettack', seed=None):
        self.name = name.lower()
        self.setting = setting.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs'], \
            'Currently only support cora, citeseer, cora_ml, polblogs'
        assert self.setting in ['gcn', 'nettack'], 'Settings should be gcn or nettack'

        self.seed = seed
        self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_filename = osp.join(root, self.name)
        self.data_filename += '.npz'

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()

    def get_train_val_test(self):

        if self.setting == 'nettack':
            return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels, seed=self.seed)
        if self.setting == 'gcn':
            return get_train_val_test_gcn(self.labels, seed=self.seed)

    def load_data(self):
        if not osp.exists(self.data_filename):
            self.download_npz()
        print('Loading {} dataset...'.format(self.name))
        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_npz(self):
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')


    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
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
        if features is None:
            features = np.eye(adj.shape[0])
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


