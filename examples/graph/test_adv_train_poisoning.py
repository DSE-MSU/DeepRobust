import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset

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

# load pre-attacked graph
perturbed_data = PtbDataset(root='/tmp/', name=args.dataset)
perturbed_adj = perturbed_data.adj

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
    adversary.attack(perturbed_adj, n_perturbations=n_perturbations, type='remove')
    modified_adj = adversary.modified_adj
    model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

model.eval()

# test directly or fine tune
print('=== test on perturbed adj ===')
output = model.predict(features, perturbed_adj)
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))

