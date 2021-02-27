Graph Dataset 
=======================

We briefly introduce the dataset format of DeepRobust through self-contained examples.
In essence, DeepRobust-Graph provides the following main features:

.. contents::
    :local: 



Clean (Unattacked) Graphs for Node Classification
-----------------------
Graphs are ubiquitous data structures describing pairwise relations between entities.
A single clean graph in DeepRobust is described by an instance of :class:`deeprobust.graph.data.Dataset`, which holds the following attributes by default:

- :obj:`data.adj`: Graph adjacency matrix in scipy.sparse.csr_matrix format with shape :obj:`[num_nodes, num_nodes]`
- :obj:`data.features`: Node feature matrix with shape :obj:`[num_nodes, num_node_features]`
- :obj:`data.labels`: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape :obj:`[num_nodes, *]`
- :obj:`data.train_idx`: Array of training node indices 
- :obj:`data.val_idx`: Array of validation node indices 
- :obj:`data.test_idx`: Array of test node indices 

By default, the loaded :obj:`deeprobust.graph.data.Dataset` will select the largest connect
component of the graph, but users specify different settings by giving different parameters. 

Currently DeepRobust supports the following datasets:
:obj:`Cora`,
:obj:`Cora-ml`,
:obj:`Citeseer`,
:obj:`Pubmed`,
:obj:`Polblogs`,
:obj:`ACM`,
:obj:`BlogCatalog`,
:obj:`Flickr`,
:obj:`UAI`.

.. code-block:: python
   
   from deeprobust.graph.data import Dataset
   data = Dataset(root='/tmp/', name='cora', seed=15) 
   adj, features, labels = data.adj, data.features, data.labels
   idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

DeepRobust also provides access to Amazon and Coauthor datasets loaded from Pytorch Geometric:
:obj:`Amazon-Computers`,
:obj:`Amazon-Photo`,
:obj:`Coauthor-CS`,
:obj:`Coauthor-Physics`.

Users can also easily create their own datasets by creating a class with the following attributes: :obj:`data.adj`, :obj:`data.features`, :obj:`data.labels`, :obj:`data.train_idx`, :obj:`data.val_idx`, :obj:`data.test_idx`.

Attacked Graphs for Node Classification
-----------------------
DeepRobust provides the attacked graphs perturbed by `metattack <https://openreview.net/pdf?id=Bylnx209YX>`_ and `nettack <https://arxiv.org/abs/1805.07984>`_. The graphs are attacked using authors' Tensorflow implementation, on random split using seed 15. The download link can be found in `ProGNN code <https://github.com/ChandlerBang/Pro-GNN>`_ and the performance of various GNNs can be found in `ProGNN paper <https://arxiv.org/abs/2005.10203>`_. They are instances of :class:`deeprobust.graph.data.PrePtbDataset` with only one attribute :obj:`adj`. Hence, :class:`deeprobust.graph.data.PrePtbDataset` is often used together with :class:`deeprobust.graph.data.Dataset` to obtain node features and labels. 

For metattack, DeepRobust provides attacked graphs for Cora, Citeseer, Polblogs and Pubmed, 
and the perturbation rate can be chosen from [0.05, 0.1, 0.15, 0.2, 0.25].

.. code-block:: python
   
   from deeprobust.graph.data import Dataset, PrePtbDataset
   data = Dataset(root='/tmp/', name='cora', seed=15) # make sure random seed is set to 15, since the attacked graph are generated under seed 15
   adj, features, labels = data.adj, data.features, data.labels
   idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
   # Load meta attacked data
   perturbed_data = PrePtbDataset(root='/tmp/',
					   name='cora',
					   attack_method='meta',
					   ptb_rate=0.05)
   perturbed_adj = perturbed_data.adj

For nettack, DeepRobust provides attacked graphs for Cora, Citeseer, Polblogs and Pubmed, 
and ptb_rate indicates the number of perturbations made on each node. 
It can be chosen from [1.0, 2.0, 3.0, 4.0, 5.0].

.. code-block:: python

   from deeprobust.graph.data import Dataset, PrePtbDataset
   data = Dataset(root='/tmp/', name='cora', seed=15) # make sure random seed is set to 15, since the attacked graph are generated under seed 15
   adj, features, labels = data.adj, data.features, data.labels
   idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
   # Load meta attacked data
   perturbed_data = PrePtbDataset(root='/tmp/', name='cora',
					   attack_method='nettack',
					   ptb_rate=3.0) # here ptb_rate means number of perturbation per nodes
   perturbed_adj = perturbed_data.adj
   idx_test = perturbed_data.target_nodes



Converting Graph Data between DeepRobust and PyTorch Geometric 
-----------------------
Given the popularity of PyTorch Geometric in the graph representation learning community,
we also provide tools for converting data between DeepRobust and PyTorch Geometric. We can
use :class:`deeprobust.graph.data.Dpr2Pyg` to convert DeepRobust data to PyTorch Geometric 
and use :class:`deeprobust.graph.data.Pyg2Dpr` to convert Pytorch Geometric data to DeepRobust.
For example, we can first create an instance of the Dataset class and convert it to pytorch geometric data format.

.. code-block:: python

    from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
    data = Dataset(root='/tmp/', name='cora') # load clean graph
    pyg_data = Dpr2Pyg(data) # convert dpr to pyg
    print(pyg_data)
    print(pyg_data[0])
    dpr_data = Pyg2Dpr(pyg_data) # convert pyg to dpr
    print(dpr_data.adj)

Load Pytorch Geometric Amazon and Coauthor Datasets
-----------------------
DeepRobust also provides access to the Amazon datasets and Coauthor datasets, i.e.,
`Amazon-Computers`, `Amazon-Photo`, `Coauthor-CS`, `Coauthor-Physics`, from Pytorch 
Geometric. Specifically, users can access them through 
:class:`deeprobust.graph.data.AmazonPyg` and :class:`deeprobust.graph.data.CoauthorPyg`. 
For example, we can directly load Amazon dataset from deeprobust in the format of pyg
as follows,

.. code-block:: python

    from deeprobust.graph.data import AmazonPyg
    computers = AmazonPyg(root='/tmp', name='computers')
    print(computers)
    print(computers[0])
    photo = AmazonPyg(root='/tmp', name='photo')
    print(photo)
    print(photo[0])


Similarly, we can also load Coauthor dataset,

.. code-block:: python

    from deeprobust.graph.data import CoauthorPyg
    cs = CoauthorPyg(root='/tmp', name='cs')
    print(cs)
    print(cs[0])
    physics = CoauthorPyg(root='/tmp', name='physics')
    print(physics)
    print(physics[0])


