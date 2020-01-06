import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from DeepRobust.image.attack.base_attack import BaseAttack

import ipdb

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
                   self.clip_max,
                   self.clip_min,
                   self.num_steps,
                   self.step_size) 
                   ##default parameter for mnist data set.

    def parse_params(self,
                     epsilon = 0.03,
                     num_steps = 40,
                     step_size = 0.01,
                     clip_max = 1.0,
                     clip_min = 0.0
                     ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.clip_max = clip_max
        self.clip_min = clip_min
        return True

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):
    print(X)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way 
    device = X.device
    imageArray = X.cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for _ in range(num_steps):

        # opt = optim.SGD([X_pgd], lr=1e-3)
        # opt.zero_grad()

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        #print(loss)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd - X.data, -epsilon, epsilon)
        print(eta)
        X_pgd = X_pgd + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        #print(X_pgd)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd