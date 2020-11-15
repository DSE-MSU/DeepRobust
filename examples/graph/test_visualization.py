from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.visualization import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
clean_adj, features = data.adj, data.features

perturbed_data = PrePtbDataset(root='/tmp/',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)
perturbed_adj = perturbed_data.adj

degree_dist(clean_adj, perturbed_adj)
feature_diff(clean_adj, perturbed_adj, features)

