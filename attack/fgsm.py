
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import linalg as LA

from attack import base_attack


class FGM(base_attack):

    def __init__(self, model, device = 'cuda'):

        super(FGM, self).__init__(model, device)


    def generate(self, input, label, **kwargs):

        ## check and parse parameters for attack
        assert self.check_type_device(input, label)
        assert self.parse_params(**kwargs)

        return fgm(self.model,
                   self.input,
                   self.label,
                   self.epsilon,
                   self.order,
                   self.clip_min,
                   self.clip_max)

    def parse_params(self,
                     epsilon = 0.1,
                     order = np.inf,
                     clip_max = None,
                     clip_min = None):
        self.epsilon = epsilon
        self.order = order
        self.clip_max = clip_max
        self.clip_min = clip_min
        return True




def fgm(model, input, label, epsilon, order, clip_min, clip_max):

    output = model(input)
    loss = F.nll_loss(output, label)

    loss.backward()
    gradient = input.grad
    gradient = np.array(gradient.cpu().detach())
    input = np.array(input.cpu().detach())

    if order == np.inf:
        d = epsilon * np.sign(gradient)
    elif order ==2:
        d = epsilon * gradient/LA.norm(gradient)
    else:
        raise ValueError('Other p norms may need other algorithms')

    x_adv = input + d

    if clip_max == None and clip_min == None:
        clip_max = np.inf
        clip_min = -np.inf

    x_adv = np.clip(x_adv, a_min=clip_min, a_max=clip_max)

    return (x_adv)


