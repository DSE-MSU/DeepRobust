import numpy as np
import scipy.sparse as sp
import time 
import copy

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree
from sklearn.cluster import KMeans
from copy import deepcopy
# from deeprobust.graph.defense_pyg import GCN, SAGE, GAT
from deeprobust.graph.targeted_attack import BaseAttack
from deeprobust.graph import utils

class UGBA(BaseAttack):
    """
    Modified from Unnoticeable Backdoor Attacks on Graph Neural Networks (WWW 2023).

    see example in examples/graph/test_ugba.py

    Parameters
    ----------
    vs_number: int
        number of selected poisoned for training backdoor model
    
    device: str
        'cpu' or 'cuda'

    target_class: int
        the class that the attacker aim to misclassify into

    trigger_size: int
        the number of nodes in a trigger
    
    target_loss_weight: float

    homo_loss_weight: float
        the weight of homophily loss

    homo_boost_thrd: float
        the upper bound of similarity 

    train_epochs: int
        the number of epochs when training GCN encoder 
    
    trojan_epochs: int
        the number of epochs when training trigger generator


    """
    def __init__(self, data, vs_number, 
                 target_class = 0, trigger_size = 3, target_loss_weight = 1, 
                 homo_loss_weight = 100, homo_boost_thrd = 0.8, train_epochs = 200, trojan_epochs = 800, dis_weight = 1, 
                 inner = 1, thrd=0.5, lr = 0.01, hidden = 32, weight_decay = 5e-4, 
                 seed = 10, debug = True, device='cpu'):
        self.device = device
        self.data = data
        self.size = vs_number
        # self.test_model = model
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.target_loss_weight = target_loss_weight
        self.homo_loss_weight = homo_loss_weight
        self.homo_boost_thrd = homo_boost_thrd
        self.train_epochs = train_epochs
        self.trojan_epochs = trojan_epochs
        self.dis_weight = dis_weight
        self.inner = inner
        self.thrd = thrd
        self.lr = lr
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.seed = seed
        self.debug = debug
    
        # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
        self.unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
        self.idx_val = utils.index_to_mask(data.val_mask, size=data.x.shape[0])
    def attack(self, target_node, x, y, edge_index, edge_weights = None):
        '''
        inject the generated trigger to the target node (a single node)

        Parameters
        ----------
        target_node: int
            the index of target node
        x: tensor:
            features of nodes
        y: tensor:
            node labels
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        '''
        idx_target = torch.tensor([target_node])
        print(idx_target)
        if(edge_weights == None):
            edge_weights = torch.ones([edge_index.shape[1]]).to(self.device)
        x, edge_index, edge_weights, y = self.inject_trigger(idx_target, x, y, edge_index, edge_weights)
        return x, edge_index, edge_weights, y 

    def get_poisoned_graph(self):
        '''
        Obtain the poisoned training graph for training backdoor GNN
        '''
        assert self.trigger_generator, "please first use train_trigger_generator() to train trigger generator and get poisoned nodes"
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = self.trigger_generator.get_poisoned()
        # add poisoned nodes into training nodes
        idx_bkd_tn = torch.cat([self.idx_train,self.idx_attach]).to(self.device)  

        poison_data = copy.deepcopy(self.data)
        idx_val = poison_data.val_mask.nonzero().flatten()
        idx_test = poison_data.test_mask.nonzero().flatten()

        poison_data.x, poison_data.edge_index, poison_data.edge_weights, poison_data.y = poison_x, poison_edge_index, poison_edge_weights, poison_labels
        poison_data.train_mask = utils.index_to_mask(idx_bkd_tn, poison_data.x.shape[0])
        poison_data.val_mask = utils.index_to_mask(idx_val, poison_data.x.shape[0])
        poison_data.test_mask = utils.index_to_mask(idx_test, poison_data.x.shape[0])
        return poison_data
    
    def train_trigger_generator(self, idx_train, edge_index, edge_weights = None, selection_method = 'cluster', **kwargs):
        """
        Train the adpative trigger generator 
        
        Parameters
        ----------
        idx_train: tensor: 
            indexs of training nodes
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        selection method : ['none', 'cluster']
            the method to select poisoned nodes
        """
        self.idx_train = idx_train
        # self.data = data

        idx_attach = self.select_idx_attach(selection_method, edge_index, edge_weights).to(self.device)
        self.idx_attach = idx_attach
        print("idx_attach: {}".format(idx_attach))
        # train trigger generator 
        trigger_generator = Backdoor(self.target_class, self.trigger_size, self.target_loss_weight, 
                                     self.homo_loss_weight, self.homo_boost_thrd, self.trojan_epochs, 
                                     self.inner, self.thrd, self.lr, self.hidden, self.weight_decay, 
                                     self.seed, self.debug, self.device)
        self.trigger_generator = trigger_generator

        self.trigger_generator.fit(self.data.x, edge_index, edge_weights, self.data.y, idx_train,idx_attach, self.unlabeled_idx)
        return self.trigger_generator, idx_attach
    
    def inject_trigger(self, idx_attach, x, y, edge_index, edge_weights):
        """
        Attach the generated triggers with the attachde nodes
        
        Parameters
        ----------
        idx_attach: tensor: 
            indexs of to-be attached nodes
        x: tensor:
            features of nodes
        y: tensor:
            node labels
        edge_index: tensor:
            edge index of the graph
        edge_weights: tensor:
            the weights of edges
        """
        assert self.trigger_generator, "please first use train_trigger_generator() to train trigger generator"

        update_x, update_edge_index,update_edge_weights, update_y = self.trigger_generator.inject_trigger(idx_attach,x,edge_index,edge_weights,y,self.device)
        return update_x, update_edge_index,update_edge_weights, update_y

    def select_idx_attach(self, selection_method, edge_index, edge_weights = None):
        if(selection_method == 'none'):
            idx_attach = self.obtain_attach_nodes(self.unlabeled_idx,self.size)
        elif(selection_method == 'cluster'):
            idx_attach = self.cluster_selection(self.data,self.idx_train,self.idx_val,self.unlabeled_idx,self.size,edge_index,edge_weights)
            idx_attach = torch.LongTensor(idx_attach).to(self.device)
        return idx_attach
    
    def obtain_attach_nodes(self,node_idxs, size):
        ### current random to implement
        size = min(len(node_idxs),size)
        rs = np.random.RandomState(self.seed)
        choice = np.arange(len(node_idxs))
        rs.shuffle(choice)
        return node_idxs[choice[:size]]

    def cluster_selection(self,data,idx_train,idx_val,unlabeled_idx,size,edge_index,edge_weights = None):
        gcn_encoder = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=32,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=0.5,                    
                            lr=0.01,                    
                            weight_decay=5e-4,                    
                            device=self.device,
                            use_ln=False,
                            layer_norm_first=False).to(self.device) 
        t_total = time.time()
        # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, edge_index, edge_weights, data.y, idx_train, idx_val= idx_val,train_iters=self.train_epochs,verbose=True)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        seen_node_idx = torch.concat([idx_train,unlabeled_idx])
        nclass = np.unique(data.y.cpu().numpy()).shape[0]
        encoder_x = gcn_encoder.get_h(data.x, edge_index,edge_weights).clone().detach()

        kmeans = KMeans(n_clusters=nclass,random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())
        # encoder_output = gcn_encoder(data.x,train_edge_index,None)
        idx_attach = self.obtain_attach_nodes_by_cluster_degree_all(edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
        idx_attach = idx_attach[:size]
        return idx_attach
    
    def obtain_attach_nodes_by_cluster_degree_all(self,edge_index,y_pred,cluster_centers,node_idxs,x,size):
        dis_weight = self.dis_weight
        degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
        distances = [] 
        for id in range(x.shape[0]):
            tmp_center_label = y_pred[id]
            tmp_center_x = cluster_centers[tmp_center_label]

            dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
            distances.append(dis)

        distances = np.array(distances)
        print(y_pred)

        nontarget_nodes = np.where(y_pred!=self.target_class)[0]

        non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
        node_idxs = np.array(non_target_node_idxs)
        candiadate_distances = distances[node_idxs]
        candiadate_degrees = degrees[node_idxs]
        candiadate_distances = self.max_norm(candiadate_distances)
        candiadate_degrees = self.max_norm(candiadate_degrees)

        dis_score = candiadate_distances + dis_weight * candiadate_degrees
        candidate_nid_index = np.argsort(dis_score)
        sorted_node_idex = np.array(node_idxs[candidate_nid_index])
        selected_nodes = sorted_node_idex
        return selected_nodes
    
    def max_norm(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,device):
        super(HomoLoss, self).__init__()
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor: 
    def __init__(self, target_class, trigger_size, target_loss_weight, homo_loss_weight, homo_boost_thrd, trojan_epochs, inner, thrd, lr, hidden, weight_decay, seed, debug, device):
        self.device = device
        self.weights = None
        self.trigger_size = trigger_size
        self.thrd = thrd
        self.trigger_index = self.get_trigger_index(self.trigger_size)
        self.hidden = hidden
        self.target_class =target_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.trojan_epochs = trojan_epochs
        self.inner = inner
        self.seed = seed
        self.target_loss_weight = target_loss_weight
        self.homo_boost_thrd = homo_boost_thrd
        self.homo_loss_weight = homo_loss_weight
        self.debug = debug
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,y,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.thrd) # may revise the process of generate
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        # update label set
        update_y = torch.cat([y,-1*torch.ones([len(idx_attach)*self.trigger_size],dtype=torch.long,device=device)])

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights, update_y


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], self.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = self.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        
        loss_best = 1e8
        for i in range(self.trojan_epochs):
            self.trojan.train()
            for j in range(self.inner):

                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat]).detach()

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()

            
            acc_train_clean = accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = accuracy(output[idx_attach], self.labels[idx_attach])
            
            # involve unlabeled nodes in outter optimization
            self.trojan.eval()
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])

            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.thrd) # may revise the process of generate
        
            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1,features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.trigger_size).to(self.device)

            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = self.target_class
            loss_target = self.target_loss_weight *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])])
            loss_homo = 0.0

            if(self.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            update_feat,\
                                            self.homo_boost_thrd)
            
            loss_outter = loss_target + self.homo_loss_weight * loss_homo

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==self.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if self.debug and i % 10 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))
        if self.debug:
            print("load best weight based on the loss outter")
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

        # torch.cuda.empty_cache()
    def get_poisoned(self):
        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights, poison_labels = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights, self.labels, self.device)
        # poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp

class GCN_Encoder(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(GCN_Encoder, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(nfeat, nhid))
        # for _ in range(layer-2):
        #     self.convs.append(GCNConv(nhid,nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        self.body = GCN_body(nfeat, nhid, dropout, layer,device=None,use_ln=use_ln,layer_norm_first=layer_norm_first)
        self.fc = nn.Linear(nhid,nclass)

        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_weight=None):
        x = self.body(x, edge_index,edge_weight)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index,edge_weight):
        self.eval()
        x = self.body(x, edge_index,edge_weight)
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids

class GCN_body(nn.Module):
    def __init__(self,nfeat, nhid, dropout=0.5, layer=2,device=None,layer_norm_first=False,use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-1):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
    def forward(self,x, edge_index,edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln: 
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # torch.cuda.empty_cache()


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids