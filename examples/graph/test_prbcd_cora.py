from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import argparse
import torch
import deeprobust.graph.utils as utils
from deeprobust.graph.global_attack import PRBCD
from deeprobust.graph.defense_pyg import GCN, SAGE, GAT

parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.1, help='perturbation rate.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Planetoid('./', 'cora')
dataset.transform = T.NormalizeFeatures()
data = dataset[0]

### we can also attack other models such as GCN, GAT, SAGE or GPRGNN
### (models in deeprobust.graph.defense_pyg), see below
print('now we choose to attack GCN model')
model = GCN(nfeat=data.x.shape[1], nhid=32, nclass=dataset.num_classes,
            nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            device=device).to(device)
agent = PRBCD(data, model=model, device=device, epochs=50) # by default, we are attacking the GCN model
agent.pretrain_model(model) # use the function to pretrain the provided model
edge_index, edge_weight = agent.attack(ptb_rate=args.ptb_rate)

print('now we choose to attack SAGE model')
model = SAGE(nfeat=data.x.shape[1], nhid=32, nclass=dataset.num_classes,
            nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            device=device).to(device)
agent = PRBCD(data, model=model, device=device, epochs=50) # by default, we are attacking the GCN model
agent.pretrain_model(model) # use the function to pretrain the provided model
edge_index, edge_weight = agent.attack(ptb_rate=args.ptb_rate)


print('now we choose to attack GAT model')
model = GAT(nfeat=data.x.shape[1], nhid=8, heads=8, weight_decay=5e-4,
            lr=0.005, nlayers=2, nclass=dataset.num_classes,
            dropout=0.5, device=device).to(device)

agent = PRBCD(data, model=model, device=device, epochs=50) # by default, we are attacking the GCN model
agent.pretrain_model(model) # use the function to pretrain the provided model
edge_index, edge_weight = agent.attack(ptb_rate=args.ptb_rate)


