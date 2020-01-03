import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from DeepRobust.graph.defense import GCNSVD
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

# Setup Surrogate Model
model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)

model = model.to(device)

print('=== testing GCN-Jaccard on perturbed graph ===')
model.fit_(features, perturbed_adj, labels, idx_train, verbose=True)
model.eval()
output = model.test(idx_test)

