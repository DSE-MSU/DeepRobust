Using PyTorch Geometric in DeepRobust
========
DeepRobust now provides interface to convert the data between
PyTorch Geometric and DeepRobust. 

.. note::
    Before we start, make sure you have successfully installed `torch_geometric 
    <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. 
    After you install torch_geometric, please reinstall DeepRobust to activate 
    the following functions.

.. contents::
    :local: 

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

For the attacked graph :class:`deeprobust.graph.PrePtbDataset`, it only has the attribute :obj:`adj`. 
To convert it to PyTorch Geometric data format, we can first convert the clean graph to Pyg and 
then update its :obj:`edge_index`:

.. code-block:: python
    
    from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg
    data = Dataset(root='/tmp/', name='cora') # load clean graph
    pyg_data = Dpr2Pyg(data) # convert dpr to pyg
    # load perturbed graph
    perturbed_data = PrePtbDataset(root='/tmp/',
            name='cora',
            attack_method='meta',
            ptb_rate=0.05)
    perturbed_adj = perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj) # inplace operation

Now :obj:`pyg_data` becomes the perturbed data in the format of PyTorch Geometric. 
We can then use it as the input for various Pytorch Geometric models!

Load OGB Datasets 
-----------------------
`Open Graph Benchmark (OGB) <https://ogb.stanford.edu/>`_ has provided various benchmark
datasets. DeepRobsut now provides interface to convert OGB dataset format (Pyg data format) 
to DeepRobust format.

.. code-block:: python

    from ogb.nodeproppred import PygNodePropPredDataset
    from deeprobust.graph.data import Pyg2Dpr
    pyg_data = PygNodePropPredDataset(name = 'ogbn-arxiv')
    dpr_data = Pyg2Dpr(pyg_data) # convert pyg to dpr
    

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


Working on PyTorch Geometric Models
-----------
In this subsection, we provide examples for using GNNs based on
PyTorch Geometric. Spefically, we use GAT :class:`deeprobust.graph.defense.GAT` and 
ChebNet :class:`deeprobust.graph.defense.ChebNet` to further illustrate (while :class:`deeprobust.graph.defense.SGC` is also available in this library).
Basically, we can first convert the DeepRobust data to PyTorch Geometric 
data and then train Pyg models.

.. code-block:: python

    from deeprobust.graph.data import Dataset, Dpr2Pyg, PrePtbDataset
    from deeprobust.graph.defense import GAT
    data = Dataset(root='/tmp/', name='cora', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    gat.fit(pyg_data, patience=100, verbose=True) # train with earlystopping
    gat.test() # test performance on clean graph 

    # load perturbed graph
    perturbed_data = PrePtbDataset(root='/tmp/',
            name='cora',
            attack_method='meta',
            ptb_rate=0.05)
    perturbed_adj = perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj) # inplace operation
    gat.fit(pyg_data, patience=100, verbose=True) # train with earlystopping
    gat.test() # test performance on perturbed graph 


.. code-block:: python

    from deeprobust.graph.data import Dataset, Dpr2Pyg
    from deeprobust.graph.defense import ChebNet
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    cheby = ChebNet(nfeat=features.shape[1],
              nhid=16, num_hops=3,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    cheby = cheby.to('cpu')
    pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    cheby.fit(pyg_data, patience=10, verbose=True) # train with earlystopping
    cheby.test()


More Details 
-----------------------
More details can be found in  
`test_gat.py <https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph/test_gat.py>`_, `test_chebnet.py <https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph/test_chebnet.py>`_ and `test_sgc.py <https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph/test_sgc.py>`_.
