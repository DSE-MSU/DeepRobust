
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import linalg as LA

from deeprobust.image.attack.base_attack import BaseAttack

class FGSM(BaseAttack):

    def __init__(self, model, device = 'cuda'):

        super(FGSM, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        label = label.type(torch.FloatTensor)

        ## check and parse parameters for attack
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return fgm(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.order,
                   self.clip_min,
                   self.clip_max,
                   self.device)

    def parse_params(self,
                     epsilon = 0.2,
                     order = np.inf,
                     clip_max = None,
                     clip_min = None):
        self.epsilon = epsilon
        self.order = order
        self.clip_max = clip_max
        self.clip_min = clip_min
        return True


def fgm(model, image, label, epsilon, order, clip_min, clip_max, device):

    imageArray = image.cpu().detach().numpy()
    X_fgsm = torch.tensor(imageArray).to(device)

    #print(image.data)

    X_fgsm.requires_grad = True

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    loss = nn.CrossEntropyLoss()(model(X_fgsm), label)

    loss.backward()
    #print(X_fgsm)
    #print(X_fgsm.grad)
    if order == np.inf:
        d = epsilon * X_fgsm.grad.data.sign()    
    elif order == 2:
        gradient = X_fgsm.grad
        d = torch.zeros(gradient.shape, device = device)
        for i in range(gradient.shape[0]):
            norm_grad = gradient[i].data/LA.norm(gradient[i].data.cpu().numpy())
            d[i] = norm_grad * epsilon
    else:
        raise ValueError('Other p norms may need other algorithms')

    x_adv = X_fgsm + d

    if clip_max == None and clip_min == None:
        clip_max = np.inf
        clip_min = -np.inf

    x_adv = torch.clamp(x_adv,clip_min, clip_max)

    return x_adv


