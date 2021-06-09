import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
from deeprobust.graph.defense import SimPGCN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use data splist provided by prognn
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


print('==================')
print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
perturbed_data = PrePtbDataset(root='/tmp/',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)
perturbed_adj = perturbed_data.adj

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup Defense Model
model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)

# using validation to pick model
model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
# You can use the inner function of model to test
model.test(idx_test)

