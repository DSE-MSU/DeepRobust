Introduction to Graph Attack with Examples
=======================
In this section, we introduce the graph attack algorithms provided 
in DeepRobust. Speficailly, they can be divied into two types: 
(1) targeted attack :class:`deeprobust.graph.targeted_attack`  and 
(2) global attack :class:`deeprobust.graph.global_attack`.

.. contents::
    :local: 


Global (Untargeted) Attack for Node Classification
-----------------------
Global (untargeted) attack aims to fool GNNs into giving wrong predictions on all 
given nodes. Specifically, DeepRobust provides the following targeted
attack algorithms:

- :class:`deeprobust.graph.global_attack.Metattack`
- :class:`deeprobust.graph.global_attack.MetaApprox`
- :class:`deeprobust.graph.global_attack.DICE`
- :class:`deeprobust.graph.global_attack.MinMax`
- :class:`deeprobust.graph.global_attack.PGDAttack`
- :class:`deeprobust.graph.global_attack.NIPA`
- :class:`deeprobust.graph.global_attack.Random`
- :class:`deeprobust.graph.global_attack.NodeEmbeddingAttack`
- :class:`deeprobust.graph.global_attack.OtherNodeEmbeddingAttack`

All the above attacks except `NodeEmbeddingAttack` and `OtherNodeEmbeddingAttack` (see details 
`here <https://deeprobust.readthedocs.io/en/latest/graph/node_embedding.html>`_ ) 
take the adjacency matrix, node feature matrix and labels as input. Usually, the adjacency 
matrix is in the format of :obj:`scipy.sparse.csr_matrix` and feature matrix can either be 
:obj:`scipy.sparse.csr_matrix` or :obj:`numpy.array`. The attack algorithm
will then transfer them into :obj:`torch.tensor` inside the class. It is also fine if you
provide :obj:`torch.tensor` as input, since the algorithm can automatically deal with it. 
Now let's take a look at an example:

.. code-block:: python

    import numpy as np
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import GCN
    from deeprobust.graph.global_attack import Metattack
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    # Setup Attack Model
    model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    # Attack
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    modified_adj = model.modified_adj # modified_adj is a torch.tensor


Targeted Attack for Node Classification
-----------------------
Targeted attack aims to fool GNNs into give wrong predictions on a 
subset of nodes. Specifically, DeepRobust provides the following targeted
attack algorithms:

- :class:`deeprobust.graph.targeted_attack.Nettack`
- :class:`deeprobust.graph.targeted_attack.RLS2V`
- :class:`deeprobust.graph.targeted_attack.FGA`
- :class:`deeprobust.graph.targeted_attack.RND`
- :class:`deeprobust.graph.targeted_attack.IGAttack`

All the above attacks take the adjacency matrix, node feature matrix and labels as input.
Usually, the adjacency matrix is in the format of :obj:`scipy.sparse.csr_matrix` and feature
matrix can either be :obj:`scipy.sparse.csr_matrix` or :obj:`numpy.array`. Now let's take a look at an example:

.. code-block:: python

    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import GCN
    from deeprobust.graph.targeted_attack import Nettack
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    # Setup Attack Model
    target_node = 0
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    # Attack
    model.attack(features, adj, labels, target_node, n_perturbations=5)
    modified_adj = model.modified_adj # scipy sparse matrix
    modified_features = model.modified_features # scipy sparse matrix

Note that we also provide scripts in :download:`test_nettack.py <https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_nettack.py>`  
for selecting nodes as reported in the 
`nettack <https://arxiv.org/abs/1805.07984>`_ paper: (1) the 10 nodes 
with highest margin of classification, i.e. they are clearly correctly classified, 
(2) the 10 nodes with lowest margin (but still correctly classified) and 
(3) 20 more nodes randomly.


More Examples 
-----------------------
More examples can be found in :class:`deeprobust.graph.targeted_attack` and 
:class:`deeprobust.graph.global_attack`. You can also find examples in 
`github code examples <https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph>`_ 
and more details in `attacks table <https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph#attack-methods>`_.
