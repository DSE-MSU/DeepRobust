import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
from deeprobust.graph.utils import get_train_val_test, get_train_val_test_gcn
import zipfile

class Dataset():
    """Dataset class contains four citation network datasets "cora", "cora-ml", "citeseer" and "pubmed",
    and one blog dataset "Polblogs". Datasets "ACM", "BlogCatalog", "Flickr", "UAI",
    "Flickr" are also available. See more details in https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph#supported-datasets.
    The 'cora', 'cora-ml', 'polblogs' and 'citeseer' are downloaded from https://github.com/danielzuegner/gnn-meta-attack/tree/master/data, and 'pubmed' is from https://github.com/tkipf/gcn/tree/master/gcn/data.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be choosen from ['cora', 'citeseer', 'cora_ml', 'polblogs',
        'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr']
    setting : string
        there are two data splits settings. The 'nettack' setting follows nettack paper
        where they select the largest connected components of the graph and use 10%/10%/80%
        nodes for training/validation/test . The 'gcn' setting follows gcn paper where they
        use 20 samples in each class for traing, 500 nodes for validation, and 1000
        nodes for test. (Note here 'gcn' setting is not a fixed split, i.e.,
        different random seed would return different data splits)
    seed : int
        random seed for splitting training/validation/test.
    require_mask : bool
        setting require_mask True to get training, validation and test mask
        (self.train_mask, self.val_mask, self.test_mask)

    Examples
    --------
	We can first create an instance of the Dataset class and then take out its attributes.

	>>> from deeprobust.graph.data import Dataset
	>>> data = Dataset(root='/tmp/', name='cora', seed=15)
	>>> adj, features, labels = data.adj, data.features, data.labels
	>>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    """

    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False):
        self.name = name.lower()
        self.setting = setting.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs',
                'pubmed', 'acm', 'blogcatalog', 'uai', 'flickr'], \
                'Currently only support cora, citeseer, cora_ml, ' + \
                'polblogs, pubmed, acm, blogcatalog, flickr'
        assert self.setting in ['gcn', 'nettack'], 'Settings should be gcn or nettack'

        self.seed = seed
        # self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_train_val_test(self):
        """Get training, validation, test splits according to self.setting (either 'nettack' or 'gcn').
        """
        if self.setting == 'nettack':
            return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels, seed=self.seed)
        if self.setting == 'gcn':
            return get_train_val_test_gcn(self.labels, seed=self.seed)

    def load_data(self):
        print('Loading {} dataset...'.format(self.name))
        if self.name == 'pubmed':
            return self.load_pubmed()

        if self.name in ['acm', 'blogcatalog', 'uai', 'flickr']:
            return self.load_zip()

        if not osp.exists(self.data_filename):
            self.download_npz()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_npz(self):
        """Download adjacen matrix npz file from self.url.
        """
        print('Downloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        try:
            print('Downloading', url)
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_zip(self, name):
        url = 'https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/other_datasets/{}.zip'.\
                format(name)
        try:
            print('Downlading', url)
            urllib.request.urlretrieve(url, osp.join(self.root, name+'.zip'))
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def load_zip(self):
        data_filename = self.data_folder + '.zip'
        name = self.name
        if not osp.exists(data_filename):
            self.download_zip(name)
            with zipfile.ZipFile(data_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root)

        feature_path = osp.join(self.data_folder, '{0}.feature'.format(name))
        label_path = osp.join(self.data_folder, '{0}.label'.format(name))
        graph_path = osp.join(self.data_folder, '{0}.edge'.format(name))

        f = np.loadtxt(feature_path, dtype = float)
        l = np.loadtxt(label_path, dtype = int)
        features = sp.csr_matrix(f, dtype=np.float32)
        # features = torch.FloatTensor(np.array(features.todense()))
        struct_edges = np.genfromtxt(graph_path, dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        n = features.shape[0]
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n, n), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        label = np.array(l)

        return sadj, features, label

    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)


        test_idx_file = "ind.{}.test.index".format(dataset)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

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
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def get_mask(self):
        idx_train, idx_val, idx_test = self.idx_train, self.idx_val, self.idx_test
        labels = self.onehot(self.labels)

        def get_mask(idx):
            mask = np.zeros(labels.shape[0], dtype=np.bool)
            mask[idx] = 1
            return mask

        def get_y(idx):
            mx = np.zeros(labels.shape)
            mx[idx] = labels[idx]
            return mx

        self.train_mask = get_mask(self.idx_train)
        self.val_mask = get_mask(self.idx_val)
        self.test_mask = get_mask(self.idx_test)
        self.y_train, self.y_val, self.y_test = get_y(idx_train), get_y(idx_val), get_y(idx_test)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    data = Dataset(root='/tmp/', name='flickr')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


