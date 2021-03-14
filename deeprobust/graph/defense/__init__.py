from .gcn import GCN, GraphConvolution
from .gcn_preprocess import GCNSVD, GCNJaccard
from .r_gcn import RGCN, GGCL_F, GGCL_D
from .prognn import ProGNN
from .simpgcn import SimPGCN
from .node_embedding import Node2Vec, DeepWalk
import warnings
try:
    from .gat import GAT
    from .chebnet import ChebNet
    from .sgc import SGC
except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
            "would like to use the datasets from pytorch " +
            "geometric. See details in https://pytorch-geom" +
            "etric.readthedocs.io/en/latest/notes/installation.html")

__all__ = ['GCN', 'GCNSVD', 'GCNJaccard', 'RGCN', 'ProGNN',
           'GraphConvolution', 'GGCL_F', 'GGCL_D', 'GAT',
           'ChebNet', 'SGC', 'SimPGCN', 'Node2Vec', 'DeepWalk']
