import torch
from deeprobust.graph.defense import GCN
import pickle
import os.path as osp
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
import os

def load_victim_model(data, model_name='gcn', device='cpu', file_path=None):
    """load_victim_model.

    Parameters
    ----------
    data : deeprobust.graph.Dataset
        graph data
    model_name : str
        victime model name, e.g. ('gcn', 'deepwalk') But currently it only
        supports gcn as victim model.
    device : str
        'cpu' or 'cuda'
    file_path :
        if given, the victim model will be loaded from this path.
    """

    assert model_name == 'gcn', 'Currently only support gcn as victim model...'
    if file_path is None:
        # file_path = f'results/saved_models/{data.name}/{model_name}_checkpoint'
        file_path = 'results/saved_models/{0}/{1}_checkpoint'.format(data.name, model_name)
    else:
        file_path = osp.join(file_path, '{}_checkpoint'.format(model_name))

    # Setup victim model
    if osp.exists(file_path):
        victim_model = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                    nhid=16, dropout=0.5, weight_decay=5e-4, device=device)

        victim_model.load_state_dict(torch.load(file_path, map_location=device))
        victim_model.to(device)
        victim_model.eval()
        return victim_model

    victim_model = train_victim_model(data=data, model_name=model_name,
                                        device=device, file_path=osp.dirname(file_path))
    return victim_model

def train_victim_model(data, model_name='gcn', file_path=None, device='cpu'):
    """Train the victim model (target classifer) and save the model
    Note that the attacker can only do black query to this model.
    """

    if file_path is None:
        file_path = 'results/saved_models/%s/' % data.name

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
    if not osp.exists(file_path):
        os.system('mkdir -p %s' % file_path)
    torch.save(victim_model.state_dict(), osp.join(file_path, model_name + '_checkpoint'))
    victim_model.eval()
    return victim_model

