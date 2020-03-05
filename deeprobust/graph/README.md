# Setup
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

# Full README
[click here](https://github.com/DSE-MSU/DeepRobust/edit/master/README.md)

# Attack Methods
|   Attack Methods   | Type<img width=200> | Perturbation <img width=80> | Evasion/<br>Poisoning | Apply Domain | Links |
|--------------------|------|--------------------|-------------|-------|----|
| Nettack | Targeted Attack | Structure<br>Features | Both | Node Classification | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf)|
| Mettack | Global Attack |  Structure<br>Features | Poisoning | Node Classification | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) |
| RL-S2V | Targeted Attack | Structure | Evasion |  Node Classification | [Adversarial Attack on Graph Structured Data](https://arxiv.org/pdf/1806.02371.pdf) |
| NIPA | Global Attack | Structure | Poisoning |  Node Classification | [Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach](https://arxiv.org/pdf/1909.06543.pdf) |
| FGSM | Targeted Attack<br>Global Attack | Structure<br>Features | Both | Node Classification | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf)|
| PGD, Min-max | Global Attack | Structure | Both | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|
| DICE | Global Attack | Structure | Both |  Node Classification | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) |
| RND | Targeted Attack<br>Global Attack | Structure<br>Features<br>Adding Nodes | Both | Node Classification | |


# Defense Methods
|   Defense Methods   | Defense Type | Apply Domain | Links |
|---------------------|--------------|--------------|------|
| RGCN | Gaussian | Node Classification | [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf) |
| GCN-Jaccard | Preprocessing | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf)|
| GCN-SVD | Preprocessing | Node Classification | [All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) |
| Adv-training | Adversarial Training | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|

<!--| Hidden-Adv-training | Adversarial Training | Node Classification<br>Graph Classification |[To be added]|
-->

