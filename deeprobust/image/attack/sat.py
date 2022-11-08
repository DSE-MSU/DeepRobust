import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class SAT(BaseAttack):
    """
    SAT attack.
    """

    def __init__(self, model, device='cuda'):

        super(SAT, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate SAT adversarial examples.
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

        if self.method == "R1":
            return sat_attack_1(self.model,
                       self.image,
                       self.label,
                       self.epsilon,
                       self.num_steps,
                       self.clip_max,
                       self.clip_min,
                       self.print_process,
                       self.distance_measure,
                       self.device)
        elif self.method == "R2":
            return sat_attack_2(self.model,
                       self.image,
                       self.label,
                       self.epsilon,
                       self.eps_iter,
                       self.clip_max,
                       self.clip_min,
                       self.print_process,
                       self.distance_measure,
                       self.device)
        elif self.method == "R3":
            return sat_attack_3(self.model,
                       self.image,
                       self.label,
                       self.epsilon,
                       self.epsilon_low = epsilon_low,
                       self.epsilon_high = epsilon_high,
                       self.alpha = alpha,
                       self.clip_max,
                       self.clip_min,
                       self.print_process,
                       self.distance_measure,
                       self.device)
        else:
            raise RuntimeError('invalid regularization method')

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=1,
                     eps_iter=0.01,
                     epsilon_low=0.007,
                     epsilon_high=0.03,
                     alpha=0.007,
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
        self.eps_iter = eps_iter
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.alpha = alpha
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def sat_attack_1(model,
              x_natural,
              y,
              epsilon,
              num_steps,
              clip_max,
              clip_min,
              print_process,
              distance_measure,
              device='cuda'):
    
    model.eval()
    y = Variable(y.to(device))
    x_adv = x_natural.to(device)
    epsilon = epsilon / num_steps

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x_adv = Variable(x_adv, requires_grad=True)
            zero_gradients(x_adv) 

            out = model(x_adv)
            cost = F.cross_entropy(out, y)
            cost.backward()

            per = epsilon * torch.sign(x_adv.grad.data)
            x_adv = x_adv.data + per.to(device)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv

def sat_attack_2(model,
              x_natural,
              y,
              epsilon,
              eps_iter,
              clip_max,
              clip_min,
              print_process,
              distance_measure,
              device='cuda'):
    
    model.eval()
    B,C,H,W = x_natural.size()

    if distance_measure == 'l_inf':
        noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(B,C,H,W))).to(device)
        x_adv = torch.cat((x_natural, x_natural + eps_iter * torch.sign(noise)), 0)
        x_adv = Variable(x_adv.to(device), requires_grad=True)
        y = Variable(y)
        y = torch.cat((y, y),0)
        zero_gradients(x_adv) 

        out = model(x_adv)
        cost = F.cross_entropy(out, y)
        cost.backward()

        x_adv = x_adv.data + epsilon * torch.sign(x_adv.grad.data)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_natural, clip_min, clip_max)

    return x_adv

def sat_attack_3(model,
              x_natural,
              y,
              epsilon_low,
              epsilon_high,
              alpha,
              clip_max,
              clip_min,
              print_process,
              distance_measure,
              device='cuda'):
    
    model.eval()
    B,C,H,W = x_natural.size()

    if distance_measure == 'l_inf':
        noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0,size=(B,C,H,W))).to(device)
        x_adv = torch.clamp(x_natural + alpha * torch.sign(noise), clip_min, clip_max)
        x_adv = Variable(x_adv.to(device), requires_grad=True)
        y = Variable(y)
        zero_gradients(x_adv)

        out  = model(x_adv)
        cost = F.cross_entropy(out, y)
        cost.backward()

        loss_grad = torch.sign(x_adv.grad.data)
        adv_low = x_adv.data + epsilon_low * loss_grad
        adv_high = x_adv.data + epsilon_high * loss_grad
        x_adv = torch.cat((adv_low, adv_high),0)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_natural, clip_min, clip_max)

    return x_adv

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
    