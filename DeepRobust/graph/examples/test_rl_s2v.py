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
from DeepRobust.graph.rl.env import NodeAttakEnv, GraphNormTool, StaticGraph
from DeepRobust.graph.utils import *
from DeepRobust.graph.data import Dataset
from DeepRobust.graph.black_box import *
from DeepRobust.graph.rl.agent import Agent
from DeepRobust.graph.rl.cmd_args import cmd_args


def init_setup():
    data = Dataset(root='/tmp/', name=cmd_args.dataset, setting='gcn')

    adj, features, labels = data.adj, data.features, data.labels
    StaticGraph.graph = nx.from_scipy_sparse_matrix(adj)
    dict_of_lists = nx.to_dict_of_lists(StaticGraph.graph)

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)

    # labels = torch.LongTensor(labels)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    device = torch.device('cuda') if cmd_args.ctx == 'gpu' else 'cpu'
    # black box setting

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    victim_model = load_victim_model(data, device=device)
    setattr(victim_model, 'norm_tool',  GraphNormTool(normalize=True, gm='gcn', device=device))
    output = victim_model.predict(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return features, labels, idx_val, idx_test, victim_model, dict_of_lists, adj

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
torch.cuda.manual_seed(cmd_args.seed)

features, labels, idx_valid, idx_test, victim_model, dict_of_lists, adj = init_setup()
output = victim_model(victim_model.features, victim_model.adj_norm)
preds = output.max(1)[1].type_as(labels)
acc = preds.eq(labels).double()
acc_test = acc[idx_test]

attack_list = []
for i in range(len(idx_test)):
    # only attack those misclassifed and degree>0 nodes
    if acc_test[i] > 0 and len(dict_of_lists[idx_test[i]]):
        attack_list.append(idx_test[i])

if not cmd_args.meta_test:
    total = attack_list
    idx_valid = idx_test
else:
    total = attack_list + idx_valid

acc_test = acc[idx_valid]
meta_list = []
num_wrong = 0
for i in range(len(idx_valid)):
    if acc_test[i] > 0:
        if len(dict_of_lists[idx_valid[i]]):
            meta_list.append(idx_valid[i])
    else:
        num_wrong += 1

print( 'meta list ratio:', len(meta_list) / float(len(idx_valid)))

device = torch.device('cuda') if cmd_args.ctx == 'gpu' else 'cpu'

env = NodeAttakEnv(features, labels, total, dict_of_lists, victim_model, num_mod=cmd_args.num_mod, reward_type=cmd_args.reward_type)
agent = Agent(env, features, labels, meta_list, attack_list, dict_of_lists, num_wrong=num_wrong,
        num_mod=cmd_args.num_mod, reward_type=cmd_args.reward_type,
        batch_size=cmd_args.batch_size, save_dir=cmd_args.save_dir,
        bilin_q=cmd_args.bilin_q, embed_dim=cmd_args.latent_dim,
        mlp_hidden=cmd_args.mlp_hidden, max_lv=cmd_args.max_lv,
        gm=cmd_args.gm, device=device)


if cmd_args.phase == 'train':
    agent.train(num_steps=cmd_args.num_steps, lr=cmd_args.learning_rate)
else:
    agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
    agent.eval(training=cmd_args.phase)
