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
For a practice of deeprobust graph package, you can also refer to https://github.com/ChandlerBang/Pro-GNN.


# Full README
[click here](https://github.com/DSE-MSU/DeepRobust)

# Supported Datasets
* Cora
* Cora-ML
* Citeseer
* Pubmed
* Polblogs
* ACM: [link1](https://github.com/zhumeiqiBUPT/AM-GCN) [link2](https://github.com/Jhy1993/HAN)
* BlogCatalog: [link](https://github.com/mengzaiqiao/CAN)
* Flickr: [link](https://github.com/mengzaiqiao/CAN)
* UAI: A Unifed Weakly Supervised Framework for Community Detection and Semantic Matching. 

For more details, please take a look at [dataset.py](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py)

# Attack Methods
|   Attack Methods   | Type<img width=200> | Perturbation <img width=80> | Evasion/<br>Poisoning | Apply Domain | Paper | Code |
|--------------------|------|--------------------|-------------|-------|----|----|
| Nettack | Targeted Attack | Structure<br>Features | Both | Node Classification | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf)| [test_nettack.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_nettack.py) |
| FGA | Targeted Attack | Structure | Both | Node Classification | [Fast Gradient Attack on Network Embedding](https://arxiv.org/pdf/1809.02797.pdf)| [test_fga.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_fga.py) |
| Metattack | Global Attack |  Structure<br>Features | Poisoning | Node Classification | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) | [test_mettack.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_mettack.py) |
| RL-S2V | Targeted Attack | Structure | Evasion |  Node Classification | [Adversarial Attack on Graph Structured Data](https://arxiv.org/pdf/1806.02371.pdf) |[test_rl_s2v.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_rl_s2v.py) |
| PGD, Min-max | Global Attack | Structure | Both | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|[test_pgd.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_pgd.py)  [test_min_max.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_min_max.py) |
| DICE | Global Attack | Structure | Both |  Node Classification | [Hiding individuals and communities in a social network](https://arxiv.org/abs/1608.00375)|[test_dice.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_dice.py) |
| IG-Attack | Targeted Attack | Structure<br>Features| Both | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf)|[test_ig.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_ig.py) |
| NIPA | Global Attack | Structure | Poisoning |  Node Classification | [Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach](https://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf) | [test_nipa.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_nipa.py) |
| RND | Targeted Attack<br>Global Attack | Structure<br>Features<br>Adding Nodes | Both | Node Classification | |[test_rnd.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_rnd.py) |

# Defense Methods
|   Defense Methods   | Defense Type | Apply Domain | Paper | Code |
|---------------------|--------------|--------------|------| ------|
| GCN | Victim Model | Node Classification | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | [test_gcn.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_gcn.py) |
| RGCN | Adaptive Aggregation | Node Classification | [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf) | [test_rgcn.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_rgcn.py) |
| GCN-Jaccard | Graph Purifying | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf)| [test_gcn_jaccard.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_gcn_jaccard.py) |
| GCN-SVD | Graph Purifying | Node Classification | [All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) | [test_gcn_svd.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_gcn_svd.py) |
| Adv-training | Adversarial Training | Node Classification |  |[test_adv_train_poisoning.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_adv_train_poisoning.py) |
| Pro-GNN | Graph Purifying | Node Classification | [Graph Structure Learning for Robust Graph Neural Network](https://arxiv.org/abs/2005.10203)|[test_prognn.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_prognn.py) |
| SimP-GCN | Adaptive Aggregation | Node Classification | [Node Similarity Preserving Graph Convolutional Networks](https://arxiv.org/abs/2011.09643)|[test_simpgcn.py](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_simpgcn.py) |
<!--| Adv-training | Adversarial Training | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf)|
-->
<!--| Hidden-Adv-training | Adversarial Training | Node Classification<br>Graph Classification |[To be added]|
-->

