import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from deeprobust.graph.rl.nipa_env import NodeInjectionEnv, GraphNormTool, StaticGraph
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.black_box import *
from deeprobust.graph.global_attack import NIPA
from deeprobust.graph.rl.nipa_config import args
import warnings


def add_nodes(self, features, adj, labels, idx_train, target_node, n_added=1, n_perturbations=10):
    print('number of pertubations: %s' % n_perturbations)
    N = adj.shape[0]
    D = features.shape[1]
    modified_adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    modified_features = self.reshape_mx(features, shape=(N+n_added, D))

    diff_labels = [l for l in range(labels.max()+1) if l != labels[target_node]]
    diff_labels = np.random.permutation(diff_labels)
    possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]

    return modified_adj, modified_features

def generate_injected_features(features, n_added):
    # TODO not sure how to generate features of injected nodes
    features = features.tolil()
    avg = np.tile(features.mean(0), (n_added, 1))
    features[-n_added: ] = avg + np.random.normal(0, 1, (n_added, features.shape[1]))
    return features

def injecting_nodes(data):
    '''
        injecting nodes to adj, features, and assign labels to the injected nodes
    '''
    adj, features, labels = data.adj, data.features, data.labels
    # features = normalize_feature(features)
    N = adj.shape[0]
    D = features.shape[1]

    n_added = int(args.ratio * N)
    print('number of injected nodes: %s' % n_added)

    data.adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    enlarged_features = reshape_mx(features, shape=(N+n_added, D))
    data.features = generate_injected_features(enlarged_features, n_added)
    data.features = normalize_feature(data.features)

    injected_labels = np.random.choice(labels.max()+1, n_added)
    data.labels = np.hstack((labels, injected_labels))

def init_setup():
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    injecting_nodes(data)

    adj, features, labels = data.adj, data.features, data.labels

    StaticGraph.graph = nx.from_scipy_sparse_matrix(adj)
    dict_of_lists = nx.to_dict_of_lists(StaticGraph.graph)

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    device = torch.device('cuda') if args.ctx == 'gpu' else 'cpu'

    # gray box setting
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    # Setup victim model
    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0.5, weight_decay=5e-4, device=device)

    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train, idx_val)
    setattr(victim_model, 'norm_tool',  GraphNormTool(normalize=True, gm='gcn', device=device))

    output = victim_model.predict(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return features, labels, idx_train, idx_val, idx_test, victim_model, dict_of_lists, adj

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

features, labels, idx_train, idx_val, idx_test, victim_model, dict_of_lists, adj = init_setup()
victim_model.eval()
output = victim_model(victim_model.features, victim_model.adj_norm)
preds = output.max(1)[1].type_as(labels)
acc = preds.eq(labels).double()
acc_test = acc[idx_test]

device = torch.device('cuda') if args.ctx == 'gpu' else 'cpu'

env = NodeInjectionEnv(features, labels, idx_train, idx_val, dict_of_lists, victim_model, ratio=args.ratio, reward_type=args.reward_type)

agent = NIPA(env, features, labels, env.idx_train, idx_val, idx_test, dict_of_lists, num_wrong=0,
        ratio=args.ratio, reward_type=args.reward_type,
        batch_size=args.batch_size, save_dir=args.save_dir,
        bilin_q=args.bilin_q, embed_dim=args.latent_dim,
        mlp_hidden=args.mlp_hidden, max_lv=args.max_lv,
        gm=args.gm, device=device)


warnings.warn("NIPA is not ready. Haven't reproduced the performance yet")
warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')

if args.phase == 'train':
    agent.train(num_episodes=10000, lr=args.learning_rate)
else:
    agent.net.load_state_dict(torch.load(args.save_dir + '/epoch-best.model'))
    agent.eval(training=args.phase)
