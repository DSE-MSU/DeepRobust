import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import RND
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset

import argparse

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
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Attack Model
target_node = 0
model = RND()

u = 0 # node to attack
assert u in idx_unlabeled

# train surrogate model
degrees = adj.sum(0).A1
n_perturbations = int(degrees[u]) # How many perturbations to perform. Default: Degree of the node

# modified_adj = model.attack(adj, labels, idx_train, target_node, n_perturbations)
model.add_nodes(features, adj, labels, idx_train, target_node, n_added=10,
                               n_perturbations=n_perturbations)
modified_adj = model.modified_adj
modified_features = model.modified_features

def test(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('probs: {probs.detach().cpu().numpy()}')

    labels_tensor = torch.LongTensor(labels).to(device)
    loss_test = F.nll_loss(output[idx_test], labels_tensor[idx_test])
    acc_test = accuracy(output[idx_test], labels_tensor[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    print('=== testing GCN on original(clean) graph ===')
    test(adj, features, target_node)
    print('=== testing GCN on perturbed graph ===')
    if modified_features is None:
        test(modified_adj, features, target_node)
    else:
        test(modified_adj, modified_features, target_node)


if __name__ == '__main__':
    main()

