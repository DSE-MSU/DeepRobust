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
from deeprobust.graph.rl.env import NodeAttackEnv, GraphNormTool, StaticGraph
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.black_box import *
from deeprobust.graph.targeted_attack import RLS2V
from deeprobust.graph.rl.rl_s2v_config import args
import warnings


def init_setup():
    data = Dataset(root='/tmp/', name=args.dataset, setting='gcn')

    data.features = normalize_feature(data.features)
    adj, features, labels = data.adj, data.features, data.labels

    StaticGraph.graph = nx.from_scipy_sparse_matrix(adj)
    dict_of_lists = nx.to_dict_of_lists(StaticGraph.graph)

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    device = torch.device('cuda') if args.ctx == 'gpu' else 'cpu'

    # black box setting
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    victim_model = load_victim_model(data, device=device, file_path=args.saved_model)
    setattr(victim_model, 'norm_tool',  GraphNormTool(normalize=True, gm='gcn', device=device))
    output = victim_model.predict(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return features, labels, idx_val, idx_test, victim_model, dict_of_lists, adj

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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

if not args.meta_test:
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

device = torch.device('cuda') if args.ctx == 'gpu' else 'cpu'

env = NodeAttackEnv(features, labels, total, dict_of_lists, victim_model, num_mod=args.num_mod, reward_type=args.reward_type)
agent = RLS2V(env, features, labels, meta_list, attack_list, dict_of_lists, num_wrong=num_wrong,
        num_mod=args.num_mod, reward_type=args.reward_type,
        batch_size=args.batch_size, save_dir=args.save_dir,
        bilin_q=args.bilin_q, embed_dim=args.latent_dim,
        mlp_hidden=args.mlp_hidden, max_lv=args.max_lv,
        gm=args.gm, device=device)

warnings.warn('if you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')

if args.phase == 'train':
    agent.train(num_steps=args.num_steps, lr=args.learning_rate)
else:
    agent.net.load_state_dict(torch.load(args.save_dir + '/epoch-best.model'))
    agent.eval(training=args.phase)
