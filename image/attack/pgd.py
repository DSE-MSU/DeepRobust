import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from DeepRobust.image.attack.base_attack import BaseAttack

class PGD(BaseAttack):

    def __init__(self, model, device = 'cuda'):

        super(PGD, self).__init__(model, device)

    def generate(self, image, label, **kwargs):

        ## check and parse parameters for attack
        label = label.type(torch.FloatTensor)

        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return pgd_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size) 
                   ##default parameter for mnist data set.

    def parse_params(self,
                     epsilon = 0.3,
                     num_steps = 40,
                     step_size = 0.01):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        return True

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    
    X_pgd = Variable(X.data, requires_grad=True)
    X_random = torch.Tensor(X_pgd.shape).uniform_(-epsilon, epsilon).to(X_pgd.device)
    X_pgd = Variable(torch.clamp(X_pgd + X_random, 0, 1.0), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd