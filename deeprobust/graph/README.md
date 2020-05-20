# Setup
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```
# Test Examples
Test GCN on perturbed graph (5% metattack)
```
python examples/graph/test_gcn.py --dataset cora
```
Test GCN-Jaccard on perturbed graph (5% metattack)
```
python examples/graph/test_gcn_jaccard.py --dataset cora
```
Generate attack by yourself
```
python examples/graph/test_mettack.py --dataset cora --ptb_rate 0.05 
```

# Full README
[click here](https://github.com/DSE-MSU/DeepRobust)

# Attack Methods
|   Attack Methods   | Type<img width=200> | Perturbation <img width=80> | Evasion/<br>Poisoning | Apply Domain | Links |
|--------------------|------|--------------------|-------------|-------|----|
| Nettack | Targeted Attack | Structure<br>Features | Both | Node Classification | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf)|
| FGA | Targeted Attack | Structure | Both | Node Classification | [Fast Gradient Attack on Network Embedding](https://arxiv.org/pdf/1809.02797.pdf)|
| Mettack | Global Attack |  Structure<br>Features | Poisoning | Node Classification | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) |
| RL-S2V | Targeted Attack | Structure | Evasion |  Node Classification | [Adversarial Attack on Graph Structured Data](https://arxiv.org/pdf/1806.02371.pdf) |
| PGD, Min-max | Global Attack | Structure | Both | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|
| DICE | Global Attack | Structure | Both |  Node Classification | [Hiding individuals and communities in a social network](https://arxiv.org/abs/1608.00375)|
| IG-Attack | Targeted Attack | Structure<br>Features| Both | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf)|
| NIPA | Global Attack | Structure | Poisoning |  Node Classification | [Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach](https://arxiv.org/pdf/1909.06543.pdf) |
| RND | Targeted Attack<br>Global Attack | Structure<br>Features<br>Adding Nodes | Both | Node Classification | |

# Defense Methods
|   Defense Methods   | Defense Type | Apply Domain | Links |
|---------------------|--------------|--------------|------|
| RGCN | Gaussian | Node Classification | [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf) |
| GCN-Jaccard | Graph Purifying | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf)|
| GCN-SVD | Graph Purifying | Node Classification | [All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) |
| Adv-training | Adversarial Training | Node Classification | 
| Pro-GNN | Preprocessing | Node Classification | [Graph Structure Learning for Robust Graph Neural Network]()|
<!--| Adv-training | Adversarial Training | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|
-->
<!--| Hidden-Adv-training | Adversarial Training | Node Classification<br>Graph Classification |[To be added]|
-->

