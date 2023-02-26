from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import argparse
import torch
import deeprobust.graph.utils as utils
from deeprobust.graph.global_attack import PRBCD

parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.1, help='perturbation rate.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PygNodePropPredDataset(name='ogbn-arxiv')
dataset.transform = T.NormalizeFeatures()
data = dataset[0]
if not hasattr(data, 'train_mask'):
    utils.add_mask(data, dataset)

data.edge_index = to_undirected(data.edge_index, data.num_nodes)
agent = PRBCD(data, device=device)
edge_index, edge_weight = agent.attack(ptb_rate=args.ptb_rate)


