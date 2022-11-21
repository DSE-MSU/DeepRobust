import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class ADT(BaseAttack):
    """
    ADT attack.
    """

    def __init__(self, model, device='cuda'):

        super(ADT, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate ADT adversarial examples.
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

        return adt_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size,
                   self.num_samples,
                   self.lr,
                   self.lbd,
                   self.clip_max,
                   self.clip_min,
                   self.print_process,
                   self.distance_measure,
                   self.device)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     num_samples=10,
                     lr=1.0,
                     lbd=0.01,
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
        self.num_samples = num_samples
        self.lr = lr
        self.lbd = lbd
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def adt_attack(model,
              x_natural,
              y,
              epsilon,
              num_steps,
              step_size,
              num_samples,
              lr,
              lbd,
              clip_max,
              clip_min,
              print_process,
              distance_measure,
              device='cuda'):
    
    # define KL-loss
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    mean = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
    var = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)
    optimizer_adv = torch.optim.Adam([mean, var], lr=lr, betas=(0.0, 0.0))

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            for s in range(num_samples):
                adv_std = F.softplus(var)
                rand_noise = torch.randn_like(x_natural)
                adv = torch.tanh(mean + rand_noise * adv_std)

                # omit the constants in -logp
                negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
                entropy = negative_logp.mean() # entropy
                x_adv = torch.clamp(x_natural + epsilon * adv, clip_min, clip_max)

                # minimize the negative loss
                with torch.enable_grad():
                    loss = -F.cross_entropy(model(x_adv), y) - lbd * entropy
                loss.backward(retain_graph=True if s != num_samples - 1 else False)

            optimizer_adv.step()
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = torch.clamp(x_natural + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(x_natural)), clip_min, clip_max)

    return x_adv
    