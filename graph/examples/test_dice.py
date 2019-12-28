import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from DeepRobust.graph.defense import GCN
from DeepRobust.graph.global_attack import DICE
from DeepRobust.graph.utils import *

import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels = load_data(dataset=args.dataset)

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Attack Model
model = DICE()

n_perturbations = int(args.ptb_rate * adj.nnz / 2)  # How many perturbations to perform. Default: Degree of the node

modified_adj = model.attack(adj, labels, n_perturbations)

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)

modified_adj = normalize_adj(modified_adj)
modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
modified_adj = modified_adj.to(device)

def test(adj):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5)

    if args.cuda:
        gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train)

    gcn.eval()

    try:
        adj = normalize_adj_tensor(adj, sparse=True)
    except:
        adj = normalize_adj_tensor(adj)

    output = gcn(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj)


if __name__ == '__main__':
    main()

