"""
This is an implementation of [1]
References
---------
.. [1] Pang, T., Yang, X., Dong, Y., Xu, K., Zhu, J., & Su, H. (2020). 
Boosting adversarial training with hypersphere embedding. 
Advances in Neural Information Processing Systems, 33, 7779-7792.
This implementation is based on their code: https://github.com/ShawnXYang/AT_HE
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
from deeprobust.image.attack.athe import ATHE, HELoss
from deeprobust.image.attack.pgd import PGD

class ATHE(BaseDefense):
    """
    ATHE.
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
            # TRADES training
            self.train(train_loader, optimizer, epoch)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'athe_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     beta=1.0,
                     loss_func='pgd',
                     m=0.2,
                     s=15.0,
                     seed=1,
                     log_interval=100,
                     test_freq=10,
                     save_model=True,
                     save_dir='./defense_models/athe/',
                     save_freq=10,
                     clip_max=1.0,
                     clip_min=0.0,
                     distance_measure='l_inf',
                     print_process=False,
                     test_epsilon=0.031,
                     test_num_steps=20,
                     test_step_size=0.007):
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
        :param loss_func : str 
            - parameter for loss function used in training. choice=['pgd', 'trades']
        :param m : float 
            - parameter for angular margin
        :param s : float 
            - parameter for s value
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.beta = beta
        self.m=m
        self.s=s
        self.seed = seed 
        self.log_interval = log_interval
        self.test_freq = test_freq
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.distance_measure = distance_measure
        self.print_process = print_process
        self.test_epsilon = test_epsilon
        self.test_num_steps = test_num_steps
        self.test_step_size = test_step_size

    def test(self, test_loader):
        self.model.eval()

        clean_test_loss = 0
        adv_test_loss = 0
        clean_correct = 0
        adv_correct = 0

        test_adversary = PGD(self.model)
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            clean_test_loss += F.cross_entropy(clean_output, target, reduction='sum').item()
            clean_pred = clean_output.max(1, keepdim=True)[1]
            clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()

            # robust evaluation using PGD-20
            data_adv = test_adversary.generate(data, target, epsilon=self.test_epsilon, num_steps=self.test_num_steps, step_size=self.test_step_size, 
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
            data_adv = self.adv_data(data, target)
            # calculate training loss
            loss = self.calculate_loss(data, data_adv, target)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data(self, data, target):
        """
        Generate input(adversarial) data for training.
        """

        adversary = ATHE(self.model)
        data_adv = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size, 
                                      loss_func=self.loss_func, s=self.s, clip_max=self.clip_max, clip_min=self.clip_min, 
                                      print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss(self, x_natural, x_adv, y):
        """
        Calculate ATHE loss.
        """
        batch_size = len(x_natural)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        if loss_func == 'pgd':
            criterion_loss = HELoss(s=self.s) 
            # calculate robust loss
            loss = criterion_loss(self.model(x_adv), y, cm=self.m)
        elif loss_func == 'trades':
            criterion_loss = nn.KLDivLoss(reduction='sum')
            natural_loss = HELoss(s=self.s)

            # calculate nature loss
            logits = self.model(x_natural)
            loss_natural = natural_loss(logits, y, cm=self.m)
            # calculate robust loss
            loss_robust = (1.0 / batch_size) * criterion_loss(F.log_softmax(self.s * self.model(x_adv), dim=1), F.softmax(self.s * logits, dim=1))
            loss = loss_natural + self.beta * loss_robust
        else:
            raise RuntimeError('invalid loss function')
        
        return loss
