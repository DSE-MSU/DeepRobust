"""
This is an implementation of pgd adversarial training.
References
----------
..[1]Mądry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
Towards Deep Learning Models Resistant to Adversarial Attacks. stat, 1050, 9.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.defense.base_defense import BaseDefense


class PGDtraining(BaseDefense):
    """
    PGD adversarial training.

    """


    def __init__(self, model, device):
        if not torch.cuda.is_available():
            print('CUDA not availiable, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device

        self.model = model

    def generate(self, train_loader, test_loader, **kwargs):
        """Call this function to generate robust model.

        Parameters
        ----------
        train_loader :
            training data loader
        test_loader :
            testing data loader
        kwargs :
            kwargs
        """
        self.parse_params(**kwargs)

        torch.manual_seed(100)
        device = torch.device(self.device)

        optimizer = optim.Adam(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma = 0.1)
        save_model = True
        for epoch in range(1, self.epoch + 1):
            print('Training epoch: ', epoch, flush = True)
            self.train(self.device, train_loader, optimizer, epoch)
            self.test(self.model, self.device, test_loader)

            if (self.save_model and epoch % self.save_per_epoch == 0):
                if os.path.isdir(str(self.save_dir)):
                    torch.save(self.model.state_dict(), str(self.save_dir) + self.save_name + '_epoch' + str(epoch) + '.pth')
                    print("model saved in " + str(self.save_dir))
                else:
                    print("make new directory and save model in " + str(self.save_dir))
                    os.mkdir('./' + str(self.save_dir))
                    torch.save(self.model.state_dict(), str(self.save_dir) + self.save_name + '_epoch' + str(epoch) + '.pth')

            scheduler.step()

        return self.model

    def parse_params(self,
                     epoch_num = 100,
                     save_dir = "./defense_models",
                     save_name = "mnist_pgdtraining_0.3",
                     save_model = True,
                     epsilon = 8.0 / 255.0,
                     num_steps = 10,
                     perturb_step_size = 0.01,
                     lr = 0.1,
                     momentum = 0.1,
                     save_per_epoch = 10):
        """Parameter parser.

        Parameters
        ----------
        epoch_num : int
            epoch
        save_dir : str
            model dir
        save_name : str
            model name
        save_model : bool
            Whether to save model
        epsilon : float
            attack constraint
        num_steps : int
            PGD attack iteration time
        perturb_step_size : float
            perturb step size
        lr : float
            learning rate for adversary training process
        momentum : float
            momentum for optimizor
        """
        self.epoch = epoch_num
        self.save_model = True
        self.save_dir = save_dir
        self.save_name = save_name
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.perturb_step_size = perturb_step_size
        self.lr = lr
        self.momentum = momentum
        self.save_per_epoch = save_per_epoch

    def train(self, device, train_loader, optimizer, epoch):
        """
        training process.

        Parameters
        ----------
        device :
            device
        train_loader :
            training data loader
        optimizer :
            optimizer
        epoch :
            training epoch
        """

        self.model.train()
        correct = 0
        bs = train_loader.batch_size
        #scheduler = StepLR(optimizer, step_size = 10, gamma = 0.5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [70], gamma = 0.1)
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)

            data_adv, output = self.adv_data(data, target, ep = self.epsilon, num_steps = self.num_steps, perturb_step_size = self.perturb_step_size)
            loss = self.calculate_loss(output, target)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            #print every 10
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(bs)))
            correct = 0

            scheduler.step()


    def test(self, model, device, test_loader):
        """
        testing process.

        Parameters
        ----------
        model :
            model
        device :
            device
        test_loader :
            testing dataloder
        """
        model.eval()

        test_loss = 0
        correct = 0
        test_loss_adv = 0
        correct_adv = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # print clean accuracy
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # print adversarial accuracy
            data_adv, output_adv = self.adv_data(data, target, ep = self.epsilon, num_steps = self.num_steps)

            test_loss_adv += self.calculate_loss(output_adv, target, redmode = 'sum').item()  # sum up batch loss
            pred_adv = output_adv.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_loss_adv /= len(test_loader.dataset)

        print('\nTest set: Clean loss: {:.3f}, Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        print('\nTest set: Adv loss: {:.3f}, Adv Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_adv, correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))

    def adv_data(self, data, output, ep = 0.3, num_steps = 10, perturb_step_size = 0.01):
        """
        Generate input(adversarial) data for training.
        """

        adversary = PGD(self.model)
        data_adv = adversary.generate(data, output.flatten(), epsilon = ep, num_steps = num_steps, step_size = perturb_step_size)
        output = self.model(data_adv)

        return data_adv, output

    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss

