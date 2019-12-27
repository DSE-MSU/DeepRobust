from torch.nn.modules.module import Module

class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='gpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

    def attack(self):
        pass



