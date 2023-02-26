import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from .base_model import BaseModel
import torch.nn as nn

class AirGNN(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, K=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None, args=None):

        super(AirGNN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.lins = nn.ModuleList([])
        self.lins.append(Linear(nfeat, nhid))
        if with_bn:
            self.bns = nn.ModuleList([])
            self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.lins.append(Linear(nhid, nhid))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
        self.lins.append(Linear(nhid, nclass))

        self.prop = AdaptiveMessagePassing(K=K, alpha=args.alpha, mode=args.model, args=args)
        print(self.prop)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.name = args.model
        self.with_bn = with_bn

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        edge_index = SparseTensor.from_edge_index(edge_index, edge_weight,
                sparse_sizes=2 * x.shape[:1]).t()
        for ii, lin in enumerate(self.lins[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if self.with_bn:
                x = self.bns[ii](x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        edge_index = SparseTensor.from_edge_index(edge_index, edge_weight,
                sparse_sizes=2 * x.shape[:1]).t()
        for ii, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.with_bn:
                x = self.bns[ii](x)
            x = F.relu(x)
        x = self.prop(x, edge_index)
        return x


class AdaptiveMessagePassing(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 alpha: float,
                 dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 args=None,
                 **kwargs):

        super(AdaptiveMessagePassing, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.node_num = node_num
        self.args = args
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, mode=None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if mode == None: mode = self.mode

        if self.K <= 0:
            return x
        hh = x

        if mode == 'MLP':
            return x

        elif mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=self.alpha)

        elif mode in ['AirGNN']:
            x = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        for k in range(K):
            x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh
        return x

    def amp_forward(self, x, hh, K, edge_index):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)             #  Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index] # score is the adaptive score in Equation (14)
        return score.unsqueeze(1) * x

    def compute_LX(self, x, edge_index, edge_weight=None):
        x = x - self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={}, lambda_amp={})'.format(self.__class__.__name__, self.K,
                                                               self.alpha, self.mode, self.dropout,
                                                               self.args.lambda_amp)


