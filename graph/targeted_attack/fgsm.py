import torch
from DeepRobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from DeepRobust.graph import utils
import torch.nn.functional as F

class FGSM(BaseAttack):

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='gpu'):

        super(FGSM, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        if self.attack_structure:
            self.adj_changes = Parameter(torch.FloatTensor(nnodes))
            self.adj_changes.data.fill_(0)

        if self.attack_features:
            pass

    def attack(self, features, adj, labels, idx_train, target_node, n_perturbations):
        # adj: sp.csr_matrix
        self.surrogate.eval()
        modified_adj = deepcopy(adj).to_dense()

        print(f'number of pertubations: {n_perturbations}')
        for i in range(n_perturbations):
            modified_row = modified_adj[target_node] + self.adj_changes
            modified_adj[target_node] = modified_row
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            all_ok = [target_node]
            if self.attack_structure:
                output = self.surrogate(features, adj_norm)
                loss = F.nll_loss(output[idx_train], labels[idx_train])
                # acc_train = accuracy(output[idx_train], labels[idx_train])
                grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]
                grad = grad * (-2*modified_row + 1)
                grad[target_node] = 0
                grad_argmax = torch.argmax(grad)

            modified_adj.data[target_node][grad_argmax] = -2*modified_row[grad_argmax] + 1

            if self.attack_features:
                pass

        return modified_adj
