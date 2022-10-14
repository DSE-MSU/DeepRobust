"""
This is an implementation of [1]
References
---------
.. [1] Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019, May). 
Theoretically Principled Trade-off between Robustness and Accuracy. 
In International Conference on Machine Learning (pp. 7472-7482).
This implementation is based on their code: https://github.com/yaodongyu/TRADES
Copyright (c) 2019 Hongyang Zhang, Yaodong Yu
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
from deeprobust.image.attack.trades import TRADES

class TRADES(BaseDefense):
    """
    TRADES.
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
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'trade_model-nn-epoch{}.pt'.format(epoch)))
                    print("model saved in " + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=100,
                     lr=0.01,
                     momentum=0.9,
                     epsilon=0.3,
                     num_steps=40,
                     step_size=0.01,
                     beta=1.0,
                     seed=1,
                     log_interval=100,
                     test_freq=1,
                     save_model=True,
                     save_dir="./defense_models/trades/",
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
        self.beta = beta
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

        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Test: Clean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        test_accuracy = correct / len(test_loader.dataset)
        
        return test_loss, test_accuracy

    def train(self, device, train_loader, optimizer, epoch):
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

        adversary = TRADES(self.model)
        data_adv = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size)

        return data_adv

    def calculate_loss(self, x_natural, x_adv, y):
        """
        Calculate TRADES loss.
        """
        batch_size = len(x_natural)
        criterion_kl = nn.KLDivLoss(size_average=False)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate nature loss
        logits = self.model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        # calculate robust loss
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.model(x_adv), dim=1), F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        
        return loss
