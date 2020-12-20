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

Example Code
============
#. Image Attack and Defense

    .. code-block:: none
       $ from deeprobust.image.attack.pgd import PGD
       $ from deeprobust.image.config import attack_params
       $ from deeprobust.image.utils import download_model
       $ import torch
       $ import deeprobust.image.netmodels.resnet as resnet

       $ URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_50.pt"
       $ download_model(URL, "$MODEL_PATH$")

       $ model = resnet.ResNet18().to('cuda')
       $ model.load_state_dict(torch.load("$MODEL_PATH$"))
       $ model.eval()

       $ transform_val = transforms.Compose([transforms.ToTensor()])
       $ test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                    transform = transform_val),
                    batch_size = 10, shuffle=True)

       $ x, y = next(iter(test_loader))
       $ x = x.to('cuda').float()

       $ adversary = PGD(model, device)
       $ Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])

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
