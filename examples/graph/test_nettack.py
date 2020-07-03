import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

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

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

# Setup Attack Model
target_node = 0
assert target_node in idx_unlabeled

model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
model = model.to(device)

def main():
    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # direct attack
    model.attack(features, adj, labels, target_node, n_perturbations)
    # # indirect attack/ influencer attack
    # model.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    print(model.structure_perturbations)
    print('=== testing GCN on original(clean) graph ===')
    test(adj, features, target_node)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, modified_features, target_node)

def test(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

def multi_test_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt/num))

def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def multi_test_evasion():
    # test on 40 nodes on evasion attack
    target_gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    target_gcn = target_gcn.to(device)

    target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(target_gcn)
    num = len(node_list)

    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features

        acc = single_test(modified_adj, modified_features, target_node, gcn=target_gcn)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt/num))

if __name__ == '__main__':
    main()
    multi_test_poison()
    multi_test_evasion()


