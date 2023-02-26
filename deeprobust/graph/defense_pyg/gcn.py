import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from .base_model import BaseModel
from torch_sparse import coalesce, SparseTensor, matmul


class GCN(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(nfeat, nclass, bias=with_bias))
        else:
            self.layers.append(GCNConv(nfeat, nhid, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(nhid, nhid, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GCNConv(nhid, nclass, bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GCN'

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
        return x

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # from deeprobust.graph.defense import GCN
    data = Dataset(root='/tmp/', name='citeseer', setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCN(nfeat=features.shape[1],
          nhid=16,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cuda')
    model = model.to('cuda')
    pyg_data = Dpr2Pyg(data)[0]

    # model.fit(features, adj, labels, idx_train, train_iters=200, verbose=True)
    # model.test(idx_test)

    from utils import get_dataset
    pyg_data = get_dataset('citeseer', True, if_dpr=False)[0]

    import ipdb
    ipdb.set_trace()

    model.fit(pyg_data, verbose=True) # train with earlystopping
    model.test()
    print(model.predict())
