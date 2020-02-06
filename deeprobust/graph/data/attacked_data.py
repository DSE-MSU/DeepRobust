import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request

class PtbDataset:
    '''
        This class manages pre-attacked/perturbed adjacency matrix on different datasets
    '''

    def __init__(self, root, name, attack_method='mettack', require_lcc=True, transform=None):
        assert attack_method == 'mettack', \
            'Currently the database only stores graphs perturbed by 5% mettack'

        self.name = name.lower()
        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs'], \
            'Currently only support cora, citeseer, cora_ml, polblogs'

        self.attack_method = attack_method
        self.url = 'https://raw.githubusercontent.com/ChandlerBang/pytorch-gnn-meta-attack/master/pre-attacked/{}_{}_0.05.npz'.format(self.name, self.attack_method)
        self.root = osp.expanduser(osp.normpath(root))
        self.data_filename = osp.join(root,
                '{}_{}_0.05.npz'.format(self.name, self.attack_method))
        self.adj = self.load_data()

    def load_data(self):
        if not osp.exists(self.data_filename):
            self.download_npz()
        print('Loading {} dataset perturbed by 0.05 mettack...'.format(self.name))
        adj = sp.load_npz(self.data_filename)
        print('''UserWarning: the adjacency matrix is perturbed, using the data splits under seed 15(default seed for deeprobust.graph.data.Dataset), so if you are going to verify the attacking performance, you should use the same data splits''')
        return adj

    def download_npz(self):
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception('''Download failed! Make sure you have
                    stable Internet connection and enter the right name''')


