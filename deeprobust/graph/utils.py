import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import torch.sparse as ts
import torch.nn.functional as F
import warnings

def encode_onehot(labels):
    """Convert label to onehot format.

    Parameters
    ----------
    labels : numpy.array
        node labels

    Returns
    -------
    numpy.array
        onehot labels
    """
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx

def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.

    Parameters
    ----------
    labels : torch.LongTensor
        node labels

    Returns
    -------
    torch.LongTensor
        onehot labels tensor

    """

    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)

def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)
    edge_index, edge_weight = add_self_loops(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)

def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = torch.device("cuda" if adj.is_cuda else "cpu")
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # mx = mx.dot(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def degree_normalize_sparse_tensor(adj, fill_value=1):
    """degree_normalize_sparse_tensor.
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)

    edge_index, edge_weight = add_self_loops(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight
    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)

def degree_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """

    device = torch.device("cuda" if adj.is_cuda else "cpu")
    if sparse:
        # return  degree_normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = degree_normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx

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

def loss_acc(output, labels, targets, avg_loss=True):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()[targets]
    loss = F.nll_loss(output[targets], labels[targets], reduction='mean' if avg_loss else 'none')

    if avg_loss:
        return loss, correct.sum() / len(targets)
    return loss, correct
    # correct = correct.sum()
    # return loss, correct / len(labels)

def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

	# slower version....
    # sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # indices = torch.from_numpy(
    #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # values = torch.from_numpy(sparse_mx.data)
    # shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)



def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False

def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def get_train_test(nnodes, test_size=0.8, stratify=None, seed=None):
    """This function returns training and test set without validation.
    It can be used for settings of different label rates.

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_test :
        node test indices
    """
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - test_size
    idx_train, idx_test = train_test_split(idx, random_state=None,
                                                train_size=train_size,
                                                test_size=test_size,
                                                stratify=stratify)

    return idx_train, idx_test

def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20: ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test

def get_train_test_labelrate(labels, label_rate):
    """Get train test according to given label rate.
    """
    nclass = labels.max() + 1
    train_size = int(round(len(labels) * label_rate / nclass))
    print("=== train_size = %s ===" % train_size)
    idx_train, idx_val, idx_test = get_splits_each_class(labels, train_size=train_size)
    return idx_train, idx_test

def get_splits_each_class(labels, train_size):
    """We randomly sample n instances for class, where n = train_size.
    """
    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_size])).astype(np.int)
        idx_val = np.hstack((idx_val, labels_i[train_size: 2*train_size])).astype(np.int)
        idx_test = np.hstack((idx_test, labels_i[2*train_size: ])).astype(np.int)

    return np.random.permutation(idx_train), np.random.permutation(idx_val), \
           np.random.permutation(idx_test)


def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols


def get_degree_squence(adj):
    try:
        return adj.sum(0)
    except:
        return ts.sum(adj, dim=1).to_dense()

def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])
    # original_degree_sequence = get_degree_squence(original_adjacency)
    # current_degree_sequence = get_degree_squence(modified_adjacency)
    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold

    if allowed_edges.is_cuda:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    else:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees

def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                                 + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range!=0).sum(1) + (new_in_range!=0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n

def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha =  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha =  1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha

def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll

def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.

    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.

    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]

def visualize(your_var):
    """visualize computation graph"""
    from graphviz import Digraph
    import torch
    from torch.autograd import Variable
    from torchviz import make_dot
    make_dot(your_var).view()

def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)

# def check_path(file_path):
#     if not osp.exists(file_path):
#         os.system(f'mkdir -p {file_path}')

