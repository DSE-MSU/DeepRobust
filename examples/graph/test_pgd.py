import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# data = Dataset(root='/tmp/', name=args.dataset, setting='gcn')

from torch_geometric.datasets import Planetoid
from deeprobust.graph.data import Pyg2Dpr
dataset = Planetoid('./', name=args.dataset)
data = Pyg2Dpr(dataset)

adj, features, labels = data.adj, data.features, data.labels

features = normalize_feature(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

def test(new_adj, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(features, new_adj, labels, idx_train, idx_val, patience=30) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
    else:
        gcn.eval()
        output = gcn.predict(features.to(device), new_adj.to(device)).cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    target_gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device, lr=0.01)

    target_gcn = target_gcn.to(device)
    target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    # target_gcn.fit(features, adj, labels, idx_train)

    print('=== testing GCN on clean graph ===')
    test(adj, target_gcn)

    # Setup Attack Model
    print('=== setup attack model ===')
    model = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='CE', device=device)
    model = model.to(device)

    # model.attack(features, adj, labels, idx_train, perturbations, epochs=args.epochs)
    # Here for the labels we need to replace it with predicted ones
    fake_labels = target_gcn.predict(features.to(device), adj.to(device))
    fake_labels = torch.argmax(fake_labels, 1).cpu()
    # Besides, we need to add the idx into the whole process
    idx_fake = np.concatenate([idx_train,idx_test])

    idx_others = list(set(np.arange(len(labels))) - set(idx_train))
    fake_labels = torch.cat([labels[idx_train], fake_labels[idx_others]])
    model.attack(features, adj, fake_labels, idx_fake, perturbations, epochs=args.epochs)

    print('=== testing GCN on Evasion attack ===')

    modified_adj = model.modified_adj
    test(modified_adj, target_gcn)

    # modified_features = model.modified_features
    print('=== testing GCN on Poisoning attack ===')
    test(modified_adj)

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()

