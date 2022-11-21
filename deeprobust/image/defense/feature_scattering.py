"""
This is an implementation of [1]
References
---------
.. [1] Zhang, H., & Wang, J. (2019). 
Defense against adversarial attacks using feature scattering-based adversarial training. 
Advances in Neural Information Processing Systems, 32.
This implementation is based on their code: https://github.com/Haichao-Zhang/FeatureScatter
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from deeprobust.image.defense.base_defense import BaseDefense
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.attack.feature_scattering import FeatureScattering

class FeatureScattering(BaseDefense):
    """
    FeatureScattering.
    """

    def __init__(self, model, device='cuda'):
        if not torch.cuda.is_available():
            print('CUDA not available, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)

    def generate(self, train_loader, test_loader, **kwargs):
        """Call this function to generate robust model.

        Parameters
        ----------
        train_loader :
            train_loader
        test_loader :
            test_loader
        kwargs :
            kwargs
        """
        self.parse_params(**kwargs)
        
        torch.manual_seed(self.seed)

        # initialize model, Net() can be also used here for training
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma=0.1)

        for epoch in range(1, self.epochs + 1):
            print('Training epoch: ', epoch, flush=True)
            # FeatureScattering training
            self.train(train_loader, optimizer, epoch)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'feature_scattering_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     ls_factor=0.1,
                     seed=1,
                     log_interval=100,
                     test_freq=10,
                     save_model=True,
                     save_dir='./defense_models/feature_scattering/',
                     save_freq=10,
                     clip_max=1.0,
                     clip_min=0.0,
                    ):
        """Parameter parser.
        ----------
        :param epoch : int 
            - pgd training epoch
        :param save_dir : str 
            - directory path to save model
        :param epsilon : float 
            - perturb constraint of pgd adversary example used to train defense model
        :param num_steps : int 
            - the perturb 
        :param perturb_step_size : float 
            - step_size 
        :param lr : float 
            - learning rate for adversary training process
        :param momentum : float 
            - parameter for optimizer in training process
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.ls_factor=ls_factor
        self.seed = seed 
        self.log_interval = log_interval
        self.test_freq = test_freq
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.clip_max = clip_max
        self.clip_min = clip_min

    def test(self, test_loader):
        self.model.eval()

        clean_test_loss = 0
        adv_test_loss = 0
        clean_correct = 0
        adv_correct = 0
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            clean_test_loss += F.cross_entropy(clean_output, target, reduction='sum').item()
            clean_pred = clean_output.max(1, keepdim=True)[1]
            clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()

            # robust evaluation using PGD-20
            data_adv = pgd_attack(data, target, epsilon=self.test_epsilon, num_steps=self.test_num_steps, step_size=self.test_step_size, 
                                               clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=args.distance_measure)
            adv_output = self.model(data_adv)
            adv_test_loss += F.cross_entropy(adv_output, target, reduction='sum').item()
            adv_pred = adv_output.max(1, keepdim=True)[1]
            adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()

        lean_test_loss /= len(test_loader.dataset)
        adv_test_loss /= len(test_loader.dataset)

        print('Test: Clean loss: {:.4f}, Clean accuracy: {}/{} ({:.2f}%)'.format(
        clean_test_loss, clean_correct, len(test_loader.dataset),
        100. * clean_correct / len(test_loader.dataset)), flush=True)
        print('Test: Robust loss: {:.4f}, Robust accuracy: {}/{} ({:.2f}%)'.format(
        adv_test_loss, adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)), flush=True)

    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)

            # generate adversarial examples
            data_adv, loss_fs = self.adv_data(data, target, train_flag=True)
            # calculate training loss
            loss = self.calculate_loss(loss_fs)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data(self, data, target, train_flag):
        """
        Generate input(adversarial) data for training.
        """

        adversary = FeatureScattering(self.model)
        data_adv = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size, 
                                      ls_factor=self.ls_factor, train_flag=train_flag, clip_max=self.clip_max, clip_min=self.clip_min, 
                                      print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss(self, loss_fs):
        """
        Calculate FeatureScattering loss.
        """
        loss = loss_fs.mean()
        
        return loss


def pgd_attack(model,
               x_natural,
               y,
               epsilon,
               num_steps,
               step_size,
               clip_max=1.0,
               clip_min=0.0,
               print_process=False,
               distance_measure='l_inf',
               device='cuda'):
    
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    if distance_measure == 'l_inf':
        for _ in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1), flush=True)
                
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv)[0], y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance_measure == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = torch.optim.SGD([delta], lr=epsilon / num_steps * 2)

        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1), flush=True)

            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.cross_entropy(model(adv)[0], y)
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