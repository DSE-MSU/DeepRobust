from torch.nn.modules.module import Module

class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='gpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

    def attack(self):
        pass



