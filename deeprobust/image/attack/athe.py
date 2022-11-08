import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class ATHE(BaseAttack):
    """
    ATHE attack.
    """

    def __init__(self, model, device='cuda'):

        super(ATHE, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate ATHE adversarial examples.
        Parameters
        ----------
        image :
            original image
        label :
            target label
        kwargs :
            user defined paremeters
        """

        ## check and parse parameters for attack
        label = label.type(torch.FloatTensor)

        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return athe_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size,
                   self.loss_func,
                   self.s,
                   self.clip_max,
                   self.clip_min,
                   self.print_process,
                   self.distance_measure,
                   self.device)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     loss_func='trades',
                     s=15.0,
                     clip_max=1.0,
                     clip_min=0.0,
                     print_process=False,
                     distance_measure='l_inf'):
        """parse_params.
        Parameters
        ----------
        epsilon :
            perturbation constraint
        num_steps :
            iteration step
        step_size :
            step size
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        print_process :
            whether to print out the log during optimization process, True or False print out the log during optimization process, True or False.
        distance_measure:
            distance measurement used in adversarial example generation process. choice=['l_inf', 'l_2']
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_func=loss_func
        self.s=s
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def athe_attack(model,
                  x_natural,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  loss_func,
                  s,
                  clip_max,
                  clip_min,
                  print_process,
                  distance_measure,
                  device='cuda'):
    
    # define loss
    if loss_func == 'pgd':
        criterion_loss = HELoss(s=s) 
    elif loss_func == 'trades':
        criterion_loss = nn.KLDivLoss(reduction='sum')
    else:
        raise RuntimeError('invalid loss function')

    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x_adv.requires_grad_()
            with torch.enable_grad():
                if loss_func == 'pgd':
                    loss = criterion_loss(model(x_adv), y)
                elif loss_func == 'trades':
                    loss = criterion_loss(F.log_softmax(s * model(x_adv), dim=1), F.softmax(s * model(x_natural), dim=1))
                else:
                    raise RuntimeError('invalid loss function')

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv


class HELoss(nn.Module):
    def __init__(self, s=None):
        super(HELoss, self).__init__()
        self.s = s

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        item = torch.cat([torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        loss = -torch.mean(numerator - torch.log(denominator))
        
        return loss
    