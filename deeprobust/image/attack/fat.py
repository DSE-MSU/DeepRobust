import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.image.attack.base_attack import BaseAttack

class FAT(BaseAttack):
    """
    FAT attack.
    """

    def __init__(self, model, device='cuda'):

        super(FAT, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate FAT adversarial examples.
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

        return fat_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size,
                   self.clip_max,
                   self.clip_min,
                   self.print_process,
                   self.distance_measure,
                   self.loss_fn,
                   self.tau,
                   self.omega,
                   self.rand_init,
                   self.randominit_type,
                   self.device)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     clip_max=1.0,
                     clip_min=0.0,
                     print_process=False,
                     distance_measure='l_inf',
                     loss_fn = 'cent',
                     tau=0,
                     omega=0,
                     rand_init=True,
                     randominit_type='uniform_randominit'):
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
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        self.distance_measure = distance_measure
        self.loss_fn = loss_fn
        self.tau = tau
        self.omega = omega
        self.rand_init = rand_init
        self.randominit_type = randominit_type
        
        return True

def fat_attack(model,
               x_natural,
               y,
               epsilon,
               num_steps,
               step_size,
               clip_max,
               clip_min,
               print_process,
               distance_measure,
               loss_fn,
               tau,
               omega,
               rand_init,
               randominit_type,
               device='cuda'):

    model.eval()
    output_target = []
    output_adv = []
    output_natural = []
    control = (torch.ones(len(y)) * tau).to(device)

    # initialize the adversarial data with random noise
    if rand_init:
        if randominit_type == 'normal_distribution_randominit':
            iter_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
        if randominit_type == 'uniform_randominit':
            iter_adv = x_natural.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x_natural.shape)).float().to(device)
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
    else:
        iter_adv = x_natural.to(device).detach()

    iter_clean_data = x_natural.to(device).detach()
    iter_target = y.to(device).detach()
    output_iter_clean_data = model(x_natural)

    for _ in range(num_steps):
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).to(device)
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).to(device)
                output_target = iter_target[output_index].reshape(-1).to(device)
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).to(device)), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).to(device)), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).to(device)), dim=0)

        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == 'cent':
                loss_adv = F.cross_entropy(output, iter_target)
            elif loss_fn == 'kl':
                criterion_kl = nn.KLDivLoss(size_average=False).to(device)
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
            else:
                raise RuntimeError('invalid loss function')
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().to(device)
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
        else:
            output_adv = output_adv.detach()

            return output_adv, output_target, output_natural

    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().to(device)
        output_adv = iter_adv.reshape(-1, 3, 32, 32).to(device)
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).to(device)
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).to(device)
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().to(device)
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).to(device)),dim=0).to(device)

    output_adv = output_adv.detach()

    return output_adv, output_target, output_natural
    