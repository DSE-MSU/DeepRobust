"""
Robustness of Graph Neural Networks at Scale. NeurIPS 2021.

Modified from https://github.com/sigeisler/robustness_of_gnns_at_scale/blob/main/rgnn_at_scale/attacks/prbcd.py
"""
import numpy as np
from deeprobust.graph.defense_pyg import GCN
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix


class PRBCD:

    def __init__(self, data, model=None,
            make_undirected=True,
            eps=1e-7, search_space_size=10_000_000,
            max_final_samples=20,
            fine_tune_epochs=100,
            epochs=400, lr_adj=0.1,
            with_early_stopping=True,
            do_synchronize=True,
            device='cuda',
            **kwargs
            ):
        """
        Parameters
        ----------
        data : pyg format data
        model : the model to be attacked, should be models in deeprobust.graph.defense_pyg
        """
        self.device = device
        self.data = data

        if model is None:
            model = self.pretrain_model()

        self.model = model
        nnodes = data.x.shape[0]
        d = data.x.shape[1]

        self.n, self.d = nnodes, nnodes
        self.make_undirected = make_undirected
        self.max_final_samples = max_final_samples
        self.search_space_size = search_space_size
        self.eps = eps
        self.lr_adj = lr_adj

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        # lr_factor = 0.1
        # self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs = epochs
        self.epochs_resampling = epochs - fine_tune_epochs # TODO

        self.with_early_stopping = with_early_stopping
        self.do_synchronize = do_synchronize

    def pretrain_model(self, model=None):
        data = self.data
        device = self.device
        feat, labels = data.x, data.y
        nclass = max(labels).item()+1

        if model is None:
            model = GCN(nfeat=feat.shape[1], nhid=256, dropout=0,
                    nlayers=3, with_bn=True, weight_decay=5e-4, nclass=nclass,
                    device=device).to(device)
            print(model)

        model.fit(data, train_iters=1000, patience=200, verbose=True)
        model.eval()
        model.data = data.to(self.device)
        output = model.predict()
        labels = labels.to(device)
        print(f"{model.name} Test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
        self.clean_node_mask = (output.argmax(1) == labels)
        return model


    def sample_random_block(self, n_perturbations):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations):
        best_loss = -float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        _, feat, labels = self.edge_index, self.data.x, self.data.y
        for i in range(self.max_final_samples):
            if best_loss == float('Inf') or best_loss == -float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                print(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                output = self.model.forward(feat, edge_index, edge_weight)
                loss = F.nll_loss(output[self.data.val_mask], labels[self.data.val_mask]).item()

            if best_loss < loss:
                best_loss = loss
                print('best_loss:', best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def resample_random_block(self, n_perturbations: int):
        self.keep_heuristic = 'WeightOnly'
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.search_space_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
                ] = perturbed_edge_weight_old # unique_idx: the indices for the old edges

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')


    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = -gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def _update_edge_weights(self, n_perturbations, epoch, gradient):
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(lr * gradient)
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps
        return None

    def attack(self, edge_index=None, edge_weight=None, ptb_rate=0.1):
        data = self.data
        epochs, lr_adj = self.epochs, self.lr_adj
        model = self.model
        model.eval() # should set to eval

        self.edge_index, feat, labels = data.edge_index, data.x, data.y
        with torch.no_grad():
            output = model.forward(feat, self.edge_index)
            pred = output.argmax(1)
        gt_labels = labels
        labels = labels.clone() # to avoid shallow copy
        labels[~data.train_mask] = pred[~data.train_mask]

        if edge_index is not None:
            self.edge_index = edge_index

        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(ptb_rate * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=lr_adj)
        best_loss_val = -float('Inf')
        for it in tqdm(range(epochs)):
            self.perturbed_edge_weight.requires_grad = True
            edge_index, edge_weight  = self.get_modified_adj()
            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            output = model.forward(feat, edge_index, edge_weight)
            loss = self.loss_attack(output, labels, type='tanhMargin')
            gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if it % 10 == 0:
                print(f'Epoch {it}: {loss}')

            with torch.no_grad():
                self.update_edge_weights(n_perturbations, it, gradient)
                self.perturbed_edge_weight = self.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)

                del edge_index, edge_weight #, logits

                if it < self.epochs_resampling - 1:
                    self.resample_random_block(n_perturbations)

                edge_index, edge_weight = self.get_modified_adj()
                output = model.predict(feat, edge_index, edge_weight)
                loss_val = F.nll_loss(output[data.val_mask], labels[data.val_mask])

            self.perturbed_edge_weight.requires_grad = True
            self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=lr_adj)

        # Sample final discrete graph
        edge_index, edge_weight = self.sample_final_edges(n_perturbations)
        output = model.predict(feat, edge_index, edge_weight)
        print('Test:')
        self.get_perf(output, gt_labels, data.test_mask)
        print('Validatoin:')
        self.get_perf(output, gt_labels, data.val_mask)
        return edge_index, edge_weight

    def loss_attack(self, logits, labels, type='CE'):
        self.loss_type = type
        if self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def get_perf(self, output, labels, mask, verbose=True):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        if verbose:
            print("loss= {:.4f}".format(loss.item()),
                  "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)

def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = lin_idx // n
    col_idx = lin_idx % n
    return torch.stack((row_idx, col_idx))

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))

def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu


if __name__ == "__main__":
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_undirected
    import torch_geometric.transforms as T
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0]
    if not hasattr(data, 'train_mask'):
        utils.add_mask(data, dataset)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    agent = PRBCD(data)
    edge_index, edge_weight = agent.attack()

