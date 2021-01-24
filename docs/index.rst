.. Deep documentation master file, created by
   sphinx-quickstart on Fri Jul  3 12:19:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Start building your robust models with DeepRobust!
================================
.. comments original size: 626*238

.. image:: ./DeepRobust.png
   :width: 313px
   :height: 119px

DeepRobust is a pytorch adversarial learning library, which contains most popular attack and defense algorithms in image domain and graph domain.

Installation
============
#. Activate your virtual environment

#. Install package
    .. code-block:: none
    
       $ git clone https://github.com/DSE-MSU/DeepRobust.git
       $ cd DeepRobust
       $ python setup.py install

Short Tutorial for Graph Package
========
Test your model's robustness on poisoned graph

#. Load pre-attacked graph data 
    .. code-block:: python
       
       from deeprobust.graph.data import Dataset, PrePtbDataset
       data = Dataset(root='/tmp/', name='cora', seed=15) # make sure random seed is set to 15, 
                                                          # since the attacked graph are generated under seed 15
       adj, features, labels = data.adj, data.features, data.labels
       idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
       # Load meta attacked data
       perturbed_data = PrePtbDataset(root='/tmp/',
                           name='cora',
                           attack_method='meta',
                           ptb_rate=0.05)
       perturbed_adj = perturbed_data.adj
#. Train your model on clearn/poinsed graph
    .. code-block:: python
       
       from deeprobust.graph.defense import GCN
       gcn = GCN(nfeat=features.shape[1],
           nhid=16,
           nclass=labels.max().item() + 1,
           dropout=0.5, device='cpu')
       gcn = gcn.to('cpu')
       gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train on clean graph with earlystopping
       gcn.test(idx_test)
         
       gcn.fit(features, perturbed_adj, labels, idx_train, idx_val, patience=30) # train on poisoned graph
       gcn.test(idx_test)

For more exmaples on graph package, please refer to https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph 
or https://github.com/ChandlerBang/Pro-GNN 

Example Code
============
#. Image Attack and Defense
    
    .. code-block:: python
       
       from deeprobust.image.attack.pgd import PGD
       from deeprobust.image.config import attack_params
       from deeprobust.image.utils import download_model
       import torch
       import deeprobust.image.netmodels.resnet as resnet

       URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_50.pt"
       download_model(URL, "$MODEL_PATH$")

       model = resnet.ResNet18().to('cuda')
       model.load_state_dict(torch.load("$MODEL_PATH$"))
       model.eval()

       transform_val = transforms.Compose([transforms.ToTensor()])
       test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                    transform = transform_val),
                    batch_size = 10, shuffle=True)

       x, y = next(iter(test_loader))
       x = x.to('cuda').float()

       adversary = PGD(model, device)
       Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])

    .. code-block:: python
       
       model = Net()
       train_loader = torch.utils.data.DataLoader(
                       datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
                       batch_size=100, shuffle=True)
       test_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('deeprobust/image/defense/data', train=False,
                      transform=transforms.Compose([transforms.ToTensor()])),
                      batch_size=1000,shuffle=True)
       
       defense = PGDtraining(model, 'cuda')
       defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])

Package API
===========
.. toctree::
   :maxdepth: 1
   :caption: Image Package
   
   source/deeprobust.image.attack
   source/deeprobust.image.defense
   source/deeprobust.image.netmodels


.. toctree::
   :maxdepth: 1
   :caption: Graph Package
   
   source/deeprobust.graph.global_attack
   source/deeprobust.graph.targeted_attack
   source/deeprobust.graph.defense
   source/deeprobust.graph.data

Indices and tables
==================

* :ref:`modindex`
* :ref:`search`
