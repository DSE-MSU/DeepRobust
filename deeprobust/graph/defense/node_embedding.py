"""
Code in this file is modified from https://github.com/abojchevski/node_embedding_attack

'Adversarial Attacks on Node Embeddings via Graph Poisoning'
Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
http://proceedings.mlr.press/v97/bojchevski19a.html
Copyright (C) owned by the authors, 2019
"""
import numba
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
import networkx as nx
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

class BaseEmbedding:
    """Base class for node embedding methods such as DeepWalk and Node2Vec.
    """

    def __init__(self):
        self.embedding = None
        self.model = None

    def evaluate_node_classification(self, labels, idx_train, idx_test,
            normalize_embedding=True, lr_params=None):
        """Evaluate the node embeddings on the node classification task..

        Parameters
        ---------
        labels: np.ndarray, shape [n_nodes]
            The ground truth labels
        normalize_embedding: bool
            Whether to normalize the embeddings
        idx_train: np.array
            Indices of training nodes
        idx_test: np.array
            Indices of test nodes
        lr_params: dict
            Parameters for the LogisticRegression model

        Returns
        -------
        [numpy.array, float, float] :
            Predictions from LR, micro F1 score and macro F1 score
        """

        embedding_matrix = self.embedding

        if normalize_embedding:
            embedding_matrix = normalize(embedding_matrix)

        features_train = embedding_matrix[idx_train]
        features_test = embedding_matrix[idx_test]
        labels_train = labels[idx_train]
        labels_test = labels[idx_test]

        if lr_params is None:
            lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
        else:
            lr = LogisticRegression(**lr_params)
        lr.fit(features_train, labels_train)

        lr_z_predict = lr.predict(features_test)
        f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
        f1_macro = f1_score(labels_test, lr_z_predict, average='macro')
        test_acc = accuracy_score(labels_test, lr_z_predict)
        print('Micro F1:', f1_micro)
        print('Macro F1:', f1_macro)
        return lr_z_predict, f1_micro, f1_macro


    def evaluate_link_prediction(self, adj, node_pairs, normalize_embedding=True):
        """Evaluate the node embeddings on the link prediction task.

        adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        node_pairs: numpy.array, shape [n_pairs, 2]
            Node pairs
        normalize_embedding: bool
            Whether to normalize the embeddings

        Returns
        -------
        [numpy.array, float, float]
            Inner product of embeddings, Area under ROC curve (AUC) score and average precision (AP) score
        """

        embedding_matrix = self.embedding
        if normalize_embedding:
            embedding_matrix = normalize(embedding_matrix)

        true = adj[node_pairs[:, 0], node_pairs[:, 1]].A1
        scores = (embedding_matrix[node_pairs[:, 0]] * embedding_matrix[node_pairs[:, 1]]).sum(1)
        # print(np.unique(true, return_counts=True))
        try:
            auc_score = roc_auc_score(true, scores)
        except Exception as e:
            auc_score = 0.00
            print('ROC error')
        ap_score = average_precision_score(true, scores)
        print("AUC:", auc_score)
        print("AP:", ap_score)
        return scores, auc_score, ap_score

class Node2Vec(BaseEmbedding):
    """node2vec: Scalable Feature Learning for Networks. KDD'15.
    To use this model, you need to "pip install node2vec" first.

    Examples
    ----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import NodeEmbeddingAttack
    >>> from deeprobust.graph.defense import Node2Vec
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # set up attack model
    >>> attacker = NodeEmbeddingAttack()
    >>> attacker.attack(adj, attack_type="remove", n_perturbations=1000)
    >>> modified_adj = attacker.modified_adj
    >>> print("Test Node2vec on clean graph")
    >>> model = Node2Vec()
    >>> model.fit(adj)
    >>> model.evaluate_node_classification(labels, idx_train, idx_test)
    >>> print("Test Node2vec on attacked graph")
    >>> model = Node2Vec()
    >>> model.fit(modified_adj)
    >>> model.evaluate_node_classification(labels, idx_train, idx_test)
    """

    def __init__(self):
        # self.fit = self.node2vec_snap
        super(Node2Vec, self).__init__()
        self.fit = self.node2vec

    def node2vec(self, adj, embedding_dim=64, walk_length=30, walks_per_node=10,
                      workers=8, window_size=10, num_neg_samples=1, p=4, q=1):
        """Compute Node2Vec embeddings for the given graph.

        Parameters
        ----------
        adj : sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        embedding_dim : int, optional
            Dimension of the embedding
        walks_per_node : int, optional
            Number of walks sampled from each node
        walk_length : int, optional
            Length of each random walk
        workers : int, optional
            Number of threads (see gensim.models.Word2Vec process)
        window_size : int, optional
            Window size (see gensim.models.Word2Vec)
        num_neg_samples : int, optional
            Number of negative samples (see gensim.models.Word2Vec)
        p : float
            The hyperparameter p in node2vec
        q : float
            The hyperparameter q in node2vec
        """


        walks = sample_n2v_random_walks(adj, walk_length, walks_per_node, p=p, q=q)
        walks = [list(map(str, walk)) for walk in walks]
        self.model = Word2Vec(walks, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                         iter=1, negative=num_neg_samples, hs=0, compute_loss=True)
        self.embedding = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]



class DeepWalk(BaseEmbedding):
    """DeepWalk: Online Learning of Social Representations. KDD'14. The implementation is
    modified from https://github.com/abojchevski/node_embedding_attack

    Examples
    ----
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.global_attack import NodeEmbeddingAttack
    >>> from deeprobust.graph.defense import DeepWalk
    >>> data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # set up attack model
    >>> attacker = NodeEmbeddingAttack()
    >>> attacker.attack(adj, attack_type="remove", n_perturbations=1000)
    >>> modified_adj = attacker.modified_adj
    >>> print("Test DeepWalk on clean graph")
    >>> model = DeepWalk()
    >>> model.fit(adj)
    >>> model.evaluate_node_classification(labels, idx_train, idx_test)
    >>> print("Test DeepWalk on attacked graph")
    >>> model.fit(modified_adj)
    >>> model.evaluate_node_classification(labels, idx_train, idx_test)
    >>> print("Test DeepWalk SVD")
    >>> model = DeepWalk(type="svd")
    >>> model.fit(modified_adj)
    >>> model.evaluate_node_classification(labels, idx_train, idx_test)
    """

    def __init__(self, type="skipgram"):
        super(DeepWalk, self).__init__()
        if type == "skipgram":
            self.fit = self.deepwalk_skipgram
        elif type == "svd":
            self.fit = self.deepwalk_svd
        else:
            raise NotImplementedError

    def deepwalk_skipgram(self, adj, embedding_dim=64, walk_length=80, walks_per_node=10,
                          workers=8, window_size=10, num_neg_samples=1):
        """Compute DeepWalk embeddings for the given graph using the skip-gram formulation.

        Parameters
        ----------
        adj : sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        embedding_dim : int, optional
            Dimension of the embedding
        walks_per_node : int, optional
            Number of walks sampled from each node
        walk_length : int, optional
            Length of each random walk
        workers : int, optional
            Number of threads (see gensim.models.Word2Vec process)
        window_size : int, optional
            Window size (see gensim.models.Word2Vec)
        num_neg_samples : int, optional
            Number of negative samples (see gensim.models.Word2Vec)
        """

        walks = sample_random_walks(adj, walk_length, walks_per_node)
        walks = [list(map(str, walk)) for walk in walks]
        self.model = Word2Vec(walks, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                         iter=1, negative=num_neg_samples, hs=0, compute_loss=True)
        self.embedding = self.model.wv.vectors[np.fromiter(map(int, self.model.wv.index2word), np.int32).argsort()]


    def deepwalk_svd(self, adj, window_size=10, embedding_dim=64, num_neg_samples=1, sparse=True):
        """Compute DeepWalk embeddings for the given graph using the matrix factorization formulation.
        adj: sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix of the graph
        window_size: int
            Size of the window
        embedding_dim: int
            Size of the embedding
        num_neg_samples: int
            Number of negative samples
        sparse: bool
            Whether to perform sparse operations
        Returns
        ------
        np.ndarray, shape [num_nodes, embedding_dim]
            Embedding matrix.
        """
        sum_powers_transition = sum_of_powers_of_transition_matrix(adj, window_size)

        deg = adj.sum(1).A1
        deg[deg == 0] = 1
        deg_matrix = sp.diags(1 / deg)

        volume = adj.sum()

        M = sum_powers_transition.dot(deg_matrix) * volume / (num_neg_samples * window_size)

        log_M = M.copy()
        log_M[M > 1] = np.log(log_M[M > 1])
        log_M = log_M.multiply(M > 1)

        if not sparse:
            log_M = log_M.toarray()

        Fu, Fv = self.svd_embedding(log_M, embedding_dim, sparse)

        loss = np.linalg.norm(Fu.dot(Fv.T) - log_M, ord='fro')
        self.embedding = Fu
        return Fu, Fv, loss, log_M

    def svd_embedding(self, x, embedding_dim, sparse=False):
        """Computes an embedding by selection the top (embedding_dim) largest singular-values/vectors.
        :param x: sp.csr_matrix or np.ndarray
            The matrix that we want to embed
        :param embedding_dim: int
            Dimension of the embedding
        :param sparse: bool
            Whether to perform sparse operations
        :return: np.ndarray, shape [?, embedding_dim], np.ndarray, shape [?, embedding_dim]
            Embedding matrices.
        """
        if sparse:
            U, s, V = sp.linalg.svds(x, embedding_dim)
        else:
            U, s, V = np.linalg.svd(x)

        S = np.diag(s)
        Fu = U.dot(np.sqrt(S))[:, :embedding_dim]
        Fv = np.sqrt(S).dot(V)[:embedding_dim, :].T
        return Fu, Fv


def sample_random_walks(adj, walk_length, walks_per_node, seed=None):
    """Sample random walks of fixed length from each node in the graph in parallel.
    Parameters
    ----------
    adj : sp.csr_matrix, shape [n_nodes, n_nodes]
        Sparse adjacency matrix
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    seed : int or None
        Random seed
    Returns
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        The sampled random walks
    """
    if seed is None:
        seed = np.random.randint(0, 100000)
    adj = sp.csr_matrix(adj)
    random_walks = _random_walk(adj.indptr,
                                adj.indices,
                                walk_length,
                                walks_per_node,
                                seed).reshape([-1, walk_length])
    return random_walks


@numba.jit(nopython=True, parallel=True)
def _random_walk(indptr, indices, walk_length, walks_per_node, seed):
    """Sample r random walks of length l per node in parallel from the graph.
    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    seed : int
        Random seed
    Returns
    -------
    walks : array-like, shape [r*N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for ir in range(walks_per_node):
        for n in range(N):
            for il in range(walk_length):
                walks.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])

    return np.array(walks)

def sample_n2v_random_walks(adj, walk_length, walks_per_node, p, q, seed=None):
    """Sample node2vec random walks of fixed length from each node in the graph in parallel.
    Parameters
    ----------
    adj : sp.csr_matrix, shape [n_nodes, n_nodes]
        Sparse adjacency matrix
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    p: float
        The probability to go back
    q: float,
        The probability to go explore undiscovered parts of the graphs
    seed : int or None
        Random seed
    Returns
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        The sampled random walks
    """
    if seed is None:
        seed = np.random.randint(0, 100000)
    adj = sp.csr_matrix(adj)
    random_walks = _n2v_random_walk(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q,
                                    seed)
    return random_walks

@numba.jit(nopython=True)
def random_choice(arr, p):
    """Similar to `numpy.random.choice` and it suppors p=option in numba.
    refer to <https://github.com/numba/numba/issues/2539#issuecomment-507306369>

    Parameters
    ----------
    arr : 1-D array-like
    p : 1-D array-like
        The probabilities associated with each entry in arr

    Returns
    -------
    samples : ndarray
        The generated random samples
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]

@numba.jit(nopython=True)
def _n2v_random_walk(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q,
                    seed):
    """Sample r random walks of length l per node in parallel from the graph.
    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    p: float
        The probability to go back
    q: float,
        The probability to go explore undiscovered parts of the graphs
    seed : int
        Random seed
    Returns
    -------
    walks : list generator, shape [r, N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    for _ in range(walks_per_node):
        for n in range(N):
            walk = [n]
            current_node = n
            previous_node = N
            previous_node_neighbors = np.empty(0, dtype=np.int32)
            for _ in range(walk_length - 1):
                neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                if neighbors.size == 0:
                    break

                probability = np.array([1 / q] * neighbors.size)
                probability[previous_node == neighbors] = 1 / p

                for i, nbr in enumerate(neighbors):
                    if np.any(nbr == previous_node_neighbors):
                        probability[i] = 1.

                norm_probability = probability / np.sum(probability)
                current_node = random_choice(neighbors, norm_probability)
                walk.append(current_node)
                previous_node_neighbors = neighbors
                previous_node = current_node
            yield walk

def sum_of_powers_of_transition_matrix(adj, pow):
    """Computes \sum_{r=1}^{pow) (D^{-1}A)^r.

    Parameters
    -----
    adj: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    pow: int
        Power exponent

    Returns
    ----
    sp.csr_matrix
        Sum of powers of the transition matrix of a graph.
    """
    deg = adj.sum(1).A1
    deg[deg == 0] = 1
    transition_matrix = sp.diags(1 / deg).dot(adj)

    sum_of_powers = transition_matrix
    last = transition_matrix
    for i in range(1, pow):
        last = last.dot(transition_matrix)
        sum_of_powers += last

    return sum_of_powers


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.global_attack import NodeEmbeddingAttack
    dataset_str = 'cora_ml'
    data = Dataset(root='/tmp/', name=dataset_str, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type="add_by_remove", n_perturbations=1000, n_candidates=10000)
    modified_adj = model.modified_adj

    # train defense model
    print("Test DeepWalk on clean graph")
    model = DeepWalk()
    model.fit(adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)
    # model.evaluate_node_classification(labels, idx_train, idx_test, lr_params={"max_iter": 10})

    print("Test DeepWalk on attacked graph")
    model.fit(modified_adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)
    print("\t link prediciton...")
    model.evaluate_link_prediction(modified_adj, np.array(adj.nonzero()).T)

    print("Test DeepWalk SVD")
    model = DeepWalk(type="svd")
    model.fit(modified_adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)

    # train defense model
    print("Test Node2vec on clean graph")
    model = Node2Vec()
    model.fit(adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)

    print("Test Node2vec on attacked graph")
    model = Node2Vec()
    model.fit(modified_adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)
