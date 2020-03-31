import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
import torch.nn.functional as F
import torch.optim as optim

def test(features, adj, target_node):
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
    print('probs: {}'.format(probs))
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()
