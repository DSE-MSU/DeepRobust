import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from DeepRobust.graph.defense import GCN
from DeepRobust.graph.targeted_attack import FGSM
from DeepRobust.graph.utils import *
from DeepRobust.graph.data import Dataset

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

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False)

adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
target_node = 0
model = FGSM(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
model = model.to(device)

def main():
    u = 0 # node to attack
    assert u in idx_unlabeled

    # train surrogate model

    degrees = torch.sparse.sum(adj, dim=0).to_dense()
    n_perturbations = int(degrees[u]) # How many perturbations to perform. Default: Degree of the node

    modified_adj = model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
    modified_adj = modified_adj.detach()

    print('=== testing GCN on original(clean) graph ===')
    test(adj, target_node)

    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, target_node)

def test(adj, target_node):
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

    probs = torch.exp(output[[target_node]])[0]
    print(f'probs: {probs.detach().cpu().numpy()}')
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


if __name__ == '__main__':
    main()

