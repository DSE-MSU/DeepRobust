import torch
from DeepRobust.graph.defense import GCN
import pickle
import os.path as osp
from DeepRobust.graph.data import Dataset
from DeepRobust.graph.utils import preprocess
import os

def load_victim_model(data, saved_model='gcn', device='cpu'):

    assert saved_model=='gcn', 'Currently only support gcn as victim model...'
    file_path = f'saved_models/{data.name}/{saved_model}_checkpoint'

    # Setup victim model
    if osp.exists(file_path):
        victim_model = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                    nhid=16, dropout=0.5, weight_decay=5e-4, device=device)

        victim_model.load_state_dict(torch.load(file_path, map_location=device))
        victim_model.eval()
        return victim_model

    victim_model = train_victim_model(data=data, model=saved_model,  device=device)
    return victim_model

def train_victim_model(data, model='gcn', save_dir='./saved_models', device='cpu'):
    ''' Train the victim model (target classifer) and save the model
        Note that the attacker can only do black query to this model '''

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    nfeat = features.shape[1]
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # Setup victim model
    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0.5, weight_decay=5e-4, device=device)

    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train, idx_val)

    # save the model
    file_path = f'saved_models/{data.name}/'
    if not osp.exists(file_path):
        os.system(f'mkdir -p {file_path}')
    torch.save(victim_model.state_dict(), file_path + model + '_checkpoint')
    victim_model.eval()
    return victim_model

