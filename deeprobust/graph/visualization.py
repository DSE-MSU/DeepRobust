import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import scipy.sparse as sp

def degree_dist(clean_adj, perturbed_adj, savename='degree_dist.pdf'):
    """Plot degree distributnio on clean and perturbed graphs.

    Parameters
    ----------
    clean_adj: sp.csr_matrix
        adjancecy matrix of the clean graph
    perturbed_adj: sp.csr_matrix
        adjancecy matrix of the perturbed graph
    savename: str
        filename to be saved

    Returns
    -------
    None

    """
    clean_degree = clean_adj.sum(1)
    perturbed_degree = perturbed_adj.sum(1)
    fig, ax1 = plt.subplots()
    sns.distplot(clean_degree, label='Clean Graph', norm_hist=False, ax=ax1)
    sns.distplot(perturbed_degree, label='Perturbed Graph', norm_hist=False, ax=ax1)
    ax1.grid(False)
    plt.legend(prop={'size':18})
    plt.ylabel('Density Distribution', fontsize=18)
    plt.xlabel('Node degree', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(f'Feature difference of adjacency after {attack}-attack')
    if not os.path.exists('figures/'):
       os.mkdir('figures')
    plt.savefig('figures/%s' % savename, bbox_inches='tight')
    plt.show()

def feature_diff(clean_adj, perturbed_adj, features, savename='feature_diff.pdf'):
    """Plot feature difference on clean and perturbed graphs.

    Parameters
    ----------
    clean_adj: sp.csr_matrix
        adjancecy matrix of the clean graph
    perturbed_adj: sp.csr_matrix
        adjancecy matrix of the perturbed graph
    features: sp.csr_matrix or np.array
        node features
    savename: str
        filename to be saved

    Returns
    -------
    None
    """

    fig, ax1 = plt.subplots()
    sns.distplot(_get_diff(clean_adj, features), label='Normal Edges', norm_hist=True, ax=ax1)
    delta_adj = perturbed_adj - clean_adj
    delta_adj[delta_adj < 0] = 0
    sns.distplot(_get_diff(delta_adj, features), label='Adversarial Edges', norm_hist=True, ax=ax1)
    ax1.grid(False)
    plt.legend(prop={'size':18})
    plt.ylabel('Density Distribution', fontsize=18)
    plt.xlabel('Feature Difference Between Connected Nodes', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if not os.path.exists('figures/'):
       os.mkdir('figures')
    plt.savefig('figures/%s' % savename, bbox_inches='tight')
    plt.show()


def _get_diff(adj, features):
    isSparse = sp.issparse(features)
    edges = np.array(adj.nonzero()).T
    row_degree = adj.sum(0).tolist()[0]
    diff = []
    for edge in tqdm(edges):
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue
        d = np.sum((features[n1]/np.sqrt(row_degree[n1]) - features[n2]/np.sqrt(row_degree[n2])).power(2))
        diff.append(d)
    return diff

