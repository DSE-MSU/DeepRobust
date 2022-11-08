import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class IAAT(BaseAttack):
    """
    TRADES attack.
    """

    def __init__(self, model, device='cuda'):

        super(IAAT, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate TRADES adversarial examples.
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

        return iaat_attack(self.model,
                           self.image,
                           self.label,   
                           self.num_steps,  
                           self.attack_lr_arr,
                           self.attack_eps_arr,
                           self.clip_max,
                           self.clip_min,
                           self.print_process,
                           self.distance_measure,
                           self.device)

    def parse_params(self,
                     num_steps=10,
                     attack_lr_arr=0.03,
                     attack_eps_arr=0.1,
                     clip_max=1.0,
                     clip_min=0.0,
                     print_process=False,
                     distance_measure='l_inf'):
        """parse_params.
        Parameters
        ----------
        attack_lr_arr : 
            learning rate of attacker (should be an array of batch size)
        attack_eps_arr: 
            attack epsilon of attacker (should be an array of batch size)
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        print_process :
            whether to print out the log during optimization process, True or False print out the log during optimization process, True or False.
        distance_measure:
            distance measurement used in adversarial example generation process. choice=['l_inf', 'l_2']
        """
        self.num_steps = num_steps
        self.attack_lr_arr = attack_lr_arr
        self.attack_eps_arr = attack_eps_arr
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def iaat_attack(model,
                x_natural,
                y,
                num_steps,
                attack_lr_arr,
                attack_eps_arr,
                clip_max,
                clip_min,
                print_process,
                distance_measure,
                device='cuda'):

    model.eval()

    attack_lr_arr = attack_lr_arr.view(attack_lr_arr.size(0), 1, 1, 1)
    attack_eps_arr = attack_eps_arr.view(attack_eps_arr.size(0), 1, 1, 1)
    
    # generate adversarial example
    x_adv = x_natural.detach() + (torch.randn(x_natural.shape).to(device).detach() - 0.5) * 2 * attack_eps_arr

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + attack_lr_arr * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - attack_eps_arr), x_natural + attack_eps_arr)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv
