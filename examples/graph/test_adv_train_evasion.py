import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Random
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure you use the same data splits as you generated attacks
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load original dataset (to get clean features and labels)
data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# Setup Target Model
model = GCN(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)

model = model.to(device)

# test on original adj
print('=== test on original adj ===')
model.fit(features, adj, labels, idx_train)
output = model.output
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))

print('=== Adversarial Training for Evasion Attack===')
adversary = Random()
adv_train_model = GCN(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)

adv_train_model = adv_train_model.to(device)

adv_train_model.initialize()
n_perturbations = int(0.01 * (adj.sum()//2))
for i in tqdm(range(100)):
    # modified_adj = adversary.attack(features, adj)
    adversary.attack(adj, n_perturbations=n_perturbations, type='add')
    modified_adj = adversary.modified_adj
    adv_train_model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

adv_train_model.eval()
# test directly or fine tune
print('=== test on perturbed adj ===')
output = adv_train_model.predict()
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))


# set up Surrogate & Nettack to attack the graph
import random
target_nodes = random.sample(idx_test.tolist(), 20)
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

all_margins = []
all_adv_margins = []

for target_node in target_nodes:
    # set up Nettack
    adversary = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
    adversary = adversary.to(device)
    degrees = adj.sum(0).A1
    n_perturbations = int(degrees[target_node]) + 2
    adversary.attack(features, adj, labels, target_node, n_perturbations)
    perturbed_adj = adversary.modified_adj

    model = GCN(nfeat=features.shape[1], nclass=labels.max()+1,
            nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)
    model = model.to(device)

    print('=== testing GCN on perturbed graph ===')
    model.fit(features, perturbed_adj, labels, idx_train)
    output = model.output
    margin = classification_margin(output[target_node], labels[target_node])
    all_margins.append(margin)

    print('=== testing adv-GCN on perturbed graph ===')
    output = adv_train_model.predict(features, perturbed_adj)
    adv_margin = classification_margin(output[target_node], labels[target_node])
    all_adv_margins.append(adv_margin)


print("No adversarial training: classfication margin for {0} nodes: {1}".format(len(target_nodes), np.mean(all_margins)))

print("Adversarial training: classfication margin for {0} nodes: {1}".format(len(target_nodes), np.mean(all_adv_margins)))

