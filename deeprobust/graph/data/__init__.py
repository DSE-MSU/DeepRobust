from .dataset import Dataset
from .attacked_data import PtbDataset
from .attacked_data import PrePtbDataset
import warnings
try:
    from .pyg_dataset import Pyg2Dpr, Dpr2Pyg, AmazonPyg, CoauthorPyg
except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
            "would like to use the datasets from pytorch " +
            "geometric. See details in https://pytorch-geom" +
            "etric.readthedocs.io/en/latest/notes/installation.html")


__all__ = ['Dataset', 'PtbDataset', 'PrePtbDataset',
          'Pyg2Dpr', 'Dpr2Pyg', 'AmazonPyg', 'CoauthorPyg']
