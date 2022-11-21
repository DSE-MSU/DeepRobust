import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class CAT(BaseAttack):
    """
    CAT attack.
    """

    def __init__(self, model, device='cuda'):

        super(CAT, self).__init__(model, device)

    def generate(self, image, label, label_orig, **kwargs):
        """
        Call this function to generate CAT adversarial examples.
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

        label_orig = label_orig.type(torch.FloatTensor).to(label.device)

        return cat_attack(self.model,
                   self.image,
                   self.label,
                   self.label_orig,
                   self.epsilons,
                   self.num_steps,
                   self.step_size,
                   self.loss_type,
                   self.num_classes,
                   self.kappa,
                   self.clip_max,
                   self.clip_min,
                   self.print_process,
                   self.distance_measure,
                   self.device)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     loss_type='xent',
                     num_classes=10,
                     kappa=10,
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
        self.epsilons = epsilons
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_type = loss_type
        self.num_classes=num_classes
        self.kappa=kappa
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure

        return True

def cat_attack(model,
              x_natural,
              y,
              y_orig,
              epsilons,
              num_steps,
              step_size,
              loss_type,
              num_classes,
              kappa,
              clip_max,
              clip_min,
              print_process,
              distance_measure,
              device='cuda'):
    
    # define KL-loss
    model.eval()
    batch_size = len(x_natural)
    epsilons = epsilons[:,None,None,None].repeat(1, x_natural.size(1), x_natural.size(2), x_natural.size(3))

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = loss_calc(F.log_softmax(model(x_adv), dim=1), y, y_orig, loss_type, num_classes, kappa, device)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance_measure == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for i in range(perturb_steps):
            if print_process:
                print('generating at step ' + str(i + 1))
                
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * loss_calc(F.log_softmax(model(adv), dim=1), y, y_orig, loss_type, num_classes, kappa, device)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv

def loss_calc(logits, y, y_orig, loss_type, num_classes, kappa, device):
    batch_size = len(logits)
    if loss_type == 'xent':
            probs = torch.softmax(logits, dim=1)
            loss = -torch.sum(y * torch.log(probs))/batch_size
    elif loss_type == 'mix':
        probs = torch.softmax(logits, dim=1)
        class_index = torch.arange(num_classes)[None,:].repeat(batch_size, 1).to(device)
        false_probs = torch.topk(probs[class_index!=y_orig[:,None]].view(batch_size, num_classes-1), k=1).values
        gt_probs = probs[class_index==y_orig[:,None]].unsqueeze(1)
        cw_loss = torch.max((false_probs - gt_probs).view(-1), kappa*torch.ones(batch_size).to(device))
        loss = torch.sum(torch.sum(-y * torch.log(probs), dim=1) + cw_loss)/batch_size
    else:
        raise RuntimeError('invalid loss function')

    return loss

    