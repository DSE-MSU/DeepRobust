try:
    from .gcn import GCN
    from .gat import GAT
    from .appnp import APPNP
    from .sage import SAGE
    from .gpr import GPRGNN
    from .airgnn import AirGNN
except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
                  "would like to use the datasets from pytorch " +
                  "geometric. See details in https://pytorch-geom" +
                  "etric.readthedocs.io/en/latest/notes/installation.html")

__all__ = ["GCN", "GAT", "APPNP", "SAGE", "GPRGNN", "AirGNN"]
