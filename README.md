
# DeepRobust

[Environment](#environment)

[List of Including Algorithms](#list-of-including-algorithms)

[Usage](#usage)

* [Image Attack and Defense](#image-attack-and-defense)

* [Graph Attack and Defense](#graph-attack-and-defense)
[Acknowledgement]($acknowledgement)


For more details about attacks and defenses, you can read this paper.

[Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/pdf/1909.08072.pdf)

We would be glad if you find our work useful and cite the paper.

```
@article{xu2019adversarial,
  title={Adversarial attacks and defenses in images, graphs and text: A review},
  author={Xu, Han and Ma, Yao and Liu, Haochen and Deb, Debayan and Liu, Hui and Tang, Jiliang and Jain, Anil},
  journal={arXiv preprint arXiv:1909.08072},
  year={2019}
}
```

# Environment
* `python3`
* `numpy`
* `pytorch v1.2.0`
* `torchvision v0.4.0`
* `matplotlib`

# Setup
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

# List of including algorithms
[Image](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/image)

[Graph](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph)


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
    import deeprobust.image.netmodels.resnet as resnet
    
    model = resnet.ResNet18().to('cuda')
    model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_50.pt"))
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

    More example code can be found in deeprobust/tutorials.

3. Use our evulation program to test attack algorithm against defense.

    Example:
    ```
    python -m deeprobust.image.evaluation_attack 
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
    
For more details please refer to [mettack.py](https://github.com/I-am-Bot/DeepRobust/blob/master/deeprobust/graph/examples/test_mettack.py) or run 
    ```
    python -m deeprobust.graph.examples.test_mettack.py
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
    
For more details please refer to [test_gcn_jaccard.py](https://github.com/I-am-Bot/DeepRobust/blob/master/deeprobust/graph/examples/test_gcn_jaccard.py) or run
    ```
    python -m deeprobust.graph.examples.test_gcn_jaccrad.py
    ```

## Acknowledgement
Some of the algorithms are refer to paper authors' implementations. References can be found at the top of the file. Thanks to their outstanding works!
