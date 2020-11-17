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
from deeprobust.image.utils import adjust_learning_rate

class TRADES(BaseDefense):
    """TRADES.
    """

    def __init__(self, model, device = 'cuda'):
        if not torch.cuda.is_available():
            print('CUDA not available, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)

    def generate(self, train_loader, test_loader, **kwargs):
        """generate robust model.

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

        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if (self.device == 'cuda') else {}

        # init model, Net() can be also used here for training
        optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum = self.momentum)

        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            optimizer = adjust_learning_rate(optimizer, epoch, self.lr)

            # adversarial training
            self.train(self.device, train_loader, optimizer, epoch)

            # evaluation on natural examples
            self.test(self.model, self.device, test_loader)

            # save checkpoint
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if epoch % self.save_freq == 0:
                torch.save(self.model.state_dict(),
                        os.path.join(self.save_dir, 'trade_model-nn-epoch{}.pt'.format(epoch)))
                torch.save(optimizer.state_dict(),
                        os.path.join(self.save_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
    
    def parse_params(self,
                     epochs = 100,
                     lr = 0.01,
                     momentum = 0.9,
                     epsilon = 0.3,
                     num_steps = 40,
                     step_size = 0.01,
                     beta = 1.0,
                     seed = 1,
                     log_interval = 100,
                     save_dir = "./defense_model",
                     save_freq = 10
                    ):
        """
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
        self.save_dir = save_dir
        self.save_freq = save_freq

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
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

            # calculate robust loss
            loss = self.trades_loss(model = self.model,
                            x_natural = data,
                            y = target,
                            optimizer = optimizer,
                            step_size = self.step_size,
                            epsilon = self.epsilon,
                            perturb_steps = self.num_steps,
                            beta = self.beta)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def trades_loss(self,
                model,
                x_natural,
                y,
                optimizer,
                step_size = 0.003,
                epsilon = 0.031,
                perturb_steps = 10,
                beta = 1.0,
                distance = 'l_inf'):

        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()
        batch_size = len(x_natural)
        
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        elif distance == 'l_2':
            
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
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
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(model(x_natural), dim=1))
        loss = loss_natural + beta * loss_robust
        return loss
