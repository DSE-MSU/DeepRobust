import numpy as np
import argparse
import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid

import deeprobust.graph.utils as utils
from deeprobust.graph.targeted_attack import UGBA
from deeprobust.graph.defense_pyg import GCN, SAGE, GAT

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', 
                    help='Dataset')
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=False,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=0,
                    help="number of poisoning nodes relative to the full graph")

# attack setting
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['cluster','none'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subgraph(subset, edge_index, edge_attr = None, relabel_nodes: bool = False):
        """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
        containing the nodes in :obj:`subset`.

        Args:
            subset (LongTensor, BoolTensor or [int]): The nodes to keep.
            edge_index (LongTensor): The edge indices.
            edge_attr (Tensor, optional): Edge weights or multi-dimensional
                edge features. (default: :obj:`None`)
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                :obj:`edge_index` will be relabeled to hold consecutive indices
                starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """

        device = edge_index.device

        node_mask = subset
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        return edge_index, edge_attr, edge_mask

def get_split(data, device):
    rs = np.random.RandomState(10)
    perm = rs.permutation(data.num_nodes)
    train_number = int(0.2*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(0.2*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]

    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_clean_test] = True

    return data, idx_train, idx_val, idx_clean_test, idx_atk

dataset = Planetoid('./', 'cora')
dataset.transform = T.NormalizeFeatures()
data = dataset[0]

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(data,device)
# decide clean test nodes
data.test_mask = utils.index_to_mask(idx_clean_test,size=data.x.shape[0])
data = data.to(device)
# idx_train = data.train_mask.nonzero().flatten()
# idx_val = data.val_mask.nonzero().flatten()
# idx_test = data.test_mask.nonzero().flatten()
data.edge_index = to_undirected(data.edge_index, num_nodes = data.num_nodes)
train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


test_model = GCN(nfeat=data.x.shape[1],
                    nhid=args.hidden,
                    nclass=data.y.max().item() + 1,
                    nlayers=2, lr=0.01,
                    dropout=0.5, device=device).to(device)
test_model.fit(data, train_iters=args.epochs, verbose = True)
'''get clean accuracy before attack'''
test_model.test()

'''Perform backdoor attack'''
agent = UGBA(data, vs_number = 10, device = 'cuda', trigger_size = 3,
             homo_loss_weight = 0, homo_boost_thrd = 0.5, train_epochs = 200, 
             trojan_epochs = 400, dis_weight = -1.0)
# train trigger generator
trigger_generator, idx_attach = agent.train_trigger_generator(idx_train, train_edge_index, edge_weights = None, selection_method = 'cluster')
# update poisoned training graph
poison_data = agent.get_poisoned_graph()
# train backdoored GNN
test_model = GCN(nfeat=data.x.shape[1],
                    nhid=args.hidden,
                    nclass=data.y.max().item() + 1,
                    nlayers=2, lr=0.01,
                    dropout=0.5, device=device).to(device)

# evaluation: inject trigger to target nodes
induct_data = copy.deepcopy(poison_data)
test_model.fit(induct_data, train_iters=args.epochs, verbose = True)

induct_edge_index = torch.cat([poison_data.edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_data.edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
induct_data.edge_index, induct_data.edge_weights = induct_edge_index, induct_edge_weights
acc_test = test_model.test()

'''
Attach generated trigger with a single target node: UGBA.attack(target_node, features, labels, edge_index, edge_attr)
Example: 
    x, edge_index, edge_weights, y = agent.attack(idx_atk[0], data.x, data.y, data.edge_index, None)
'''

overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
from torch_geometric.utils  import k_hop_subgraph
asr = 0
for i, idx in enumerate(idx_atk):
    idx=int(idx)
    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
    ori_node_idx = sub_induct_nodeset[sub_mapping]
    relabeled_node_idx = sub_mapping
    sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
    with torch.no_grad():
        # inject trigger on attack test nodes (idx_atk)''' 
        induct_x, induct_edge_index, induct_edge_weights, induct_y = agent.inject_trigger(idx_attach = relabeled_node_idx,x = poison_data.x[sub_induct_nodeset],y = poison_data.y[sub_induct_nodeset],edge_index = sub_induct_edge_index,edge_weights = sub_induct_edge_weights)
        induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
        # attack evaluation
        output = test_model(induct_x,induct_edge_index,induct_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
        asr += train_attach_rate
        induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
        output = output.cpu()
asr = asr/(idx_atk.shape[0])
print("Overall ASR: {:.4f}".format(asr))