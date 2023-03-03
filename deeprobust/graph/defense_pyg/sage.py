import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from .base_model import BaseModel


class SAGE(BaseModel):

    def __init__(self, nfeat, nhid, nclass, num_layers=2,
                 dropout=0.5, lr=0.01, weight_decay=0, device='cpu', with_bn=False, **kwargs):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(nfeat, nhid))

        self.bns = nn.ModuleList()
        if 'nlayers' in kwargs:
            num_layers = kwargs['nlayers']
        self.bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))

        self.convs.append(
            SAGEConv(nhid, nclass))

        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.activation = F.relu
        self.with_bn = with_bn
        self.device = device
        self.name = "SAGE"

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()

        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is not None:
                x = conv(x, adj)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is not None:
            x = self.convs[-1](x, adj)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # Deleted the following line to make propagation differentiable
        # adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
