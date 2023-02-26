from .base_attack import BaseAttack
from .dice import DICE
from .mettack import MetaApprox, Metattack
from .random_attack import Random
from .topology_attack import MinMax, PGDAttack
from .node_embedding_attack import NodeEmbeddingAttack, OtherNodeEmbeddingAttack
from .nipa import NIPA

try:
    from .prbcd import PRBCD
except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
                  "would like to use the datasets from pytorch " +
                  "geometric. See details in https://pytorch-geom" +
                  "etric.readthedocs.io/en/latest/notes/installation.html")

__all__ = ['BaseAttack', 'DICE', 'MetaApprox', 'Metattack', 'Random', 'MinMax', 'PGDAttack', 'NIPA', 'NodeEmbeddingAttack', 'OtherNodeEmbeddingAttack', 'PRBCD']
