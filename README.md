
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/rusty1s/pytorch_geometric/blob/master/CONTRIBUTING.md

<p align="center">
<img center src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/Deeprobust.png" width = "450" alt="logo">
</p>

---------------------
<!--
<a href="https://github.com/DSE-MSU/DeepRobust/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/DSE-MSU/DeepRobust"></a>  <a href="https://github.com/DSE-MSU/DeepRobust/network/members" ><img alt="GitHub forks" src="https://img.shields.io/github/forks/DSE-MSU/DeepRobust">
</a> 
-->

<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/DSE-MSU/DeepRobust"> <a href="https://github.com/DSE-MSU/DeepRobust/issues"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/DSE-MSU/DeepRobust"></a> <img alt="GitHub" src="https://img.shields.io/github/license/DSE-MSU/DeepRobust">
[![Contributing][contributing-image]][contributing-url]
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Build%20your%20robust%20machine%20learning%20models%20with%20DeepRobust%20in%2060%20seconds&url=https://github.com/DSE-MSU/DeepRobust&via=dse_msu&hashtags=MachineLearning,DeepLearning,secruity,data,developers)


<!-- <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/DSE-MSU/DeepRobust"> -->

<!--
<div align=center><img src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversarial.png" width="500"/></div>
<div align=center><img src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/graph_attack_example.png" width="00" /></div>
-->
**[Documentation](https://deeprobust.readthedocs.io/en/latest/)** | **[Paper](https://arxiv.org/abs/2005.06149)** | **[Samples](https://github.com/DSE-MSU/DeepRobust/tree/master/examples)**

DeepRobust is a PyTorch adversarial library for attack and defense methods on images and graphs. 
* If you are new to DeepRobust, we highly suggest you read the [documentation page](https://deeprobust.readthedocs.io/en/latest/) or the following content in this README to learn how to use it.  
* If you have any questions or suggestions regarding this library, feel free to create an issue [here](https://github.com/DSE-MSU/DeepRobust/issues). We will reply as soon as possible :)

<p float="left">
  <img src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/adversarial.png" width="430" />
  <img src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/graph_attack_example.png" width="380" /> 
</p>

**List of including algorithms can be found in [[Image Package]](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/image) and [[Graph Package]](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph).**

[Environment & Installation](#environment)

Usage

* [Image Attack and Defense](#image-attack-and-defense)

* [Graph Attack and Defense](#graph-attack-and-defense)

[Acknowledgement](#acknowledgement) 

For more details about attacks and defenses, you can read the following papers.
* [Adversarial Attacks and Defenses on Graphs: A Review, A Tool and Empirical Studies](https://arxiv.org/abs/2003.00653)
* [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/pdf/1909.08072.pdf)

If our work could help your research, please cite:
[DeepRobust: A PyTorch Library for Adversarial Attacks and Defenses](https://arxiv.org/abs/2005.06149)

# Changelog
* [04/2021] [Graph Package] Add support for OGB datasets.  See more details in the [tutorial page](https://deeprobust.readthedocs.io/en/latest/graph/pyg.html).
* [03/2021] [Graph Package] Added node embedding attack and victim models! See this [tutorial page](https://deeprobust.readthedocs.io/en/latest/graph/node_embedding.html).
* [02/2021] **[Graph Package] DeepRobust now provides tools for converting the datasets between [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and DeepRobust. See more details in the [tutorial page](https://deeprobust.readthedocs.io/en/latest/graph/pyg.html)!** DeepRobust now also support GAT, Chebnet and SGC based on pyg; see details in [test_gat.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_gat.py),  [test_chebnet.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_chebnet.py) and [test_sgc.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_sgc.py)
* [12/2020] DeepRobust now can be installed via pip! Try `pip install deeprobust`!
* [12/2020] [Graph Package] Add four more [datasets](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph/#supported-datasets) and one defense algorithm. More details can be found [here](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph/#defense-methods). More datasets and algorithms will be added later. Stay tuned :)
* [07/2020] Add [documentation](https://deeprobust.readthedocs.io/en/latest/) page!
* [06/2020] Add docstring to both image and graph package

# Basic Environment
* `python >= 3.6` (python 3.5 should also work)
* `pytorch >= 1.2.0`

see `setup.py` or `requirements.txt` for more information.

# Installation
## Install from pip
```
pip install deeprobust 
```
## Install from source
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

# Test Examples

```
python examples/image/test_PGD.py
python examples/image/test_pgdtraining.py
python examples/graph/test_gcn_jaccard.py --dataset cora
python examples/graph/test_mettack.py --dataset cora --ptb_rate 0.05
```

# Usage
## Image Attack and Defense
1. Train model

    Example: Train a simple CNN model on MNIST dataset for 20 epoch on gpu.
    ```python
    import deeprobust.image.netmodels.train_model as trainmodel
    trainmodel.train('CNN', 'MNIST', 'cuda', 20)
    ```
    Model would be saved in deeprobust/trained_models/.

2. Instantiated attack methods and defense methods.

    Example: Generate adversary example with PGD attack.
    ```python
    from deeprobust.image.attack.pgd import PGD
    from deeprobust.image.config import attack_params
    from deeprobust.image.utils import download_model
    import torch
    import deeprobust.image.netmodels.resnet as resnet
    from torchvision import transforms,datasets
    
    URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_20.pt"
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
    
    adversary = PGD(model, 'cuda')
    Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])
    ```

    Example: Train defense model.
    ```python
    from deeprobust.image.defense.pgdtraining import PGDtraining
    from deeprobust.image.config import defense_params
    from deeprobust.image.netmodels.CNN import Net
    import torch
    from torchvision import datasets, transforms 
    
    model = Net()
    train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()])),
                                    batch_size=100,shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                  datasets.MNIST('deeprobust/image/defense/data', train=False,
                                transform=transforms.Compose([transforms.ToTensor()])),
                                batch_size=1000,shuffle=True)

    defense = PGDtraining(model, 'cuda')
    defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])
    ```

    More example code can be found in deeprobust/examples.

3. Use our evulation program to test attack algorithm against defense.

    Example:
    ```
    cd DeepRobust
    python examples/image/test_train.py
    python deeprobust/image/evaluation_attack.py
    ```

## Graph Attack and Defense 

### Attacking Graph Neural Networks

1. Load dataset
    ```python
    import torch
    import numpy as np
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import GCN
    from deeprobust.graph.global_attack import Metattack

    data = Dataset(root='/tmp/', name='cora', setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    ```

2. Set up surrogate model
    ```python
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    ```


3. Set up attack model and generate perturbations
    ```python
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device)
    model = model.to(device)
    perturbations = int(0.05 * (adj.sum() // 2))
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    ```
    
For more details please refer to [mettack.py](https://github.com/I-am-Bot/DeepRobust/blob/master/examples/graph/test_mettack.py) or run 
    ```
    python examples/graph/test_mettack.py --dataset cora --ptb_rate 0.05
    ```

### Defending Against Graph Attacks

1. Load dataset
    ```python
    import torch
    from deeprobust.graph.data import Dataset, PtbDataset
    from deeprobust.graph.defense import GCN, GCNJaccard
    import numpy as np
    np.random.seed(15)

    # load clean graph
    data = Dataset(root='/tmp/', name='cora', setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # load pre-attacked graph by mettack
    perturbed_data = PtbDataset(root='/tmp/', name='cora')
    perturbed_adj = perturbed_data.adj
    ```
2. Test 
    ```python
    # Set up defense model and test performance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)

    # Test on GCN
    model = GCN(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)
    ```
    
For more details please refer to [test_gcn_jaccard.py](https://github.com/I-am-Bot/DeepRobust/blob/master/examples/graph/test_gcn_jaccard.py) or run
    ```
    python examples/graph/test_gcn_jaccard.py --dataset cora
    ```

## Sample Results
adversary examples generated by fgsm:
<div align="center">
<img height=140 src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/mnist_advexample_fgsm_ori.png"/><img height=140 src="https://github.com/DSE-MSU/DeepRobust/blob/master/adversary_examples/mnist_advexample_fgsm_adv.png"/>
</div>
Left:original, classified as 6; Right:adversary, classified as 4.

Serveral trained models can be found here: https://drive.google.com/open?id=1uGLiuCyd8zCAQ8tPz9DDUQH6zm-C4tEL

## Acknowledgement
Some of the algorithms are referred to paper authors' implementations. References can be found at the top of each file. 

Implementation of network structure are referred to weiaicunzai's github. Original code can be found here:
[pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)

Thanks to their outstanding works!


<!----
We would be glad if you find our work useful and cite the paper.

'''
@misc{jin2020adversarial,
    title={Adversarial Attacks and Defenses on Graphs: A Review and Empirical Study},
    author={Wei Jin and Yaxin Li and Han Xu and Yiqi Wang and Jiliang Tang},
    year={2020},
    eprint={2003.00653},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
'''
```
@article{xu2019adversarial,
  title={Adversarial attacks and defenses in images, graphs and text: A review},
  author={Xu, Han and Ma, Yao and Liu, Haochen and Deb, Debayan and Liu, Hui and Tang, Jiliang and Jain, Anil},
  journal={arXiv preprint arXiv:1909.08072},
  year={2019}
}
```
---->
