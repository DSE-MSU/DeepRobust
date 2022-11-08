import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class GAIRAT(BaseAttack):
    """
    GAIRAT attack.
    """

    def __init__(self, model, device='cuda'):

        super(GAIRAT, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate GAIRAT adversarial examples.
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

        return gairat_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size,
                   self.category,
                   self.loss_fn,
                   self.rand_init,
                   self.clip_max,
                   self.clip_min,
                   self.print_process,
                   self.distance_measure)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     category='pgd',
                     loss_fn='cent',
                     rand_init=True,
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
        self.category = category
        self.loss_fn = loss_fn
        self.rand_init = rand_init
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def gairat_attack(model,
                  x_natural,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  category,
                  loss_fn,
                  rand_init=True,
                  clip_max,
                  clip_min,
                  print_process,
                  distance_measure,
                  device='cuda'):
    
    Kappa = torch.zeros(len(x_natural))
    model.eval()
    # generate adversarial example
    if category == 'trades':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach() if rand_init else x_natural.detach()
        nat_output = model(x_natural)
    elif category == 'pgd':
        x_adv = x_natural.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x_natural.shape)).float().to(device) if rand_init else x_natural.detach()
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise RuntimeError('invalid loss function')

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x_adv.requires_grad_()
            output = model(x_adv)
            predict = output.max(1, keepdim=True)[1]
            # Update Kappa
            for p in range(len(x_adv)):
                if predict[p] == y[p]:
                    Kappa[p] += 1
            model.zero_grad()
            with torch.enable_grad():
                if loss_fn == 'cent':
                    loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, y)
                if loss_fn == 'cw':
                    loss_adv = cwloss(output, y)
                if loss_fn == 'kl':
                    criterion_kl = nn.KLDivLoss(reduction='sum').to(device)
                    loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            loss_adv.backward()
            eta = step_size * x_adv.grad.sign()
            # Update adversarial data
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv, Kappa
    