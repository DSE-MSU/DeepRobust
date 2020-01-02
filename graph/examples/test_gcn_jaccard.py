import torch
import numpy as np
import torch.nn.functional as F
from DeepRobust.graph.defense import GCNJaccard
from DeepRobust.graph.utils import *

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

adj, features, labels = load_data(dataset=args.dataset)

# shuffle
_N = adj.shape[0]
val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(_N)
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))

# Setup Surrogate Model
model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)

model = model.to(device)

print('=== testing GCN-Jaccard on perturbed graph ===')
model.fit_(features, adj, labels, idx_train)
model.eval()
output = model.test(idx_test)

