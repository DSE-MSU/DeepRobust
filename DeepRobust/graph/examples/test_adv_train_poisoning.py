import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from DeepRobust.graph.defense import GCN
from DeepRobust.graph.global_attack import Random
from DeepRobust.graph.utils import *
from DeepRobust.graph.data import Dataset
from DeepRobust.graph.data import PtbDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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

# load pre-attacked graph
perturbed_data = PtbDataset(root='/tmp/', name=args.dataset)
perturbed_adj = perturbed_data.adj

# shuffle
_N = adj.shape[0]
val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(_N)
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)

# Setup Target Model
model = GCN(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)

model = model.to(device)

adversary = Random()
# test on original adj
print('=== test on original adj ===')
model.fit(features, adj, labels, idx_train)
output = model.output
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))

print('=== testing GCN on perturbed graph ===')
model.fit(features, perturbed_adj, labels, idx_train)
output = model.output
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))


# For poisoning attack, the adjacency matrix you have
# is alreay perturbed
print('=== Adversarial Training for Poisoning Attack===')
model.initialize()
n_perturbations = int(0.01 * (adj.sum()//2))
for i in range(100):
    # modified_adj = adversary.attack(features, adj)
    modified_adj = adversary.attack(perturbed_adj, n_perturbations=n_perturbations, type='remove')
    model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

model.eval()

# test directly or fine tune
print('=== test on perturbed adj ===')
output = model.predict(features, perturbed_adj)
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))

