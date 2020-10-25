"""
This is the implementation of fgsm training.

References
 ----------
..[1]Szegedy, C., Zaremba, W., Sutskever, I., Estrach, J. B., Erhan, D., Goodfellow, I., & Fergus, R. (2014, January).
Intriguing properties of neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image
import os

from deeprobust.image.netmodels import CNN
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.defense.base_defense import BaseDefense

class FGSMtraining(BaseDefense):
    """
    FGSM adversarial training.
    """

    def __init__(self, model, device):
        if not torch.cuda.is_available():
            print('CUDA not availiable, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device

        self.model = model

    def generate(self, train_loader, test_loader, **kwargs):
        """FGSM adversarial training process.

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
        optimizer = optim.Adam(self.model.parameters(), self.lr_train)

        for epoch in range(1, self.epoch_num + 1):

            print(epoch, flush = True)
            self.train(self.device, train_loader, optimizer, epoch)
            self.test(self.model, self.device, test_loader)

            if (self.save_model):
                if os.path.isdir('./' + self.save_dir):
                    torch.save(self.model.state_dict(), './' + self.save_dir + "/" + self.save_name)
                    print("model saved in " + './' + self.save_dir)
                else:
                    print("make new directory and save model in " + './' + self.save_dir)
                    os.mkdir('./' + self.save_dir)
                    torch.save(self.model.state_dict(), './' + self.save_dir +"/" + self.save_name)

        return self.model

    def parse_params(self,
                     save_dir = "defense_models",
                     save_model = True,
                     save_name = "mnist_fgsmtraining_0.2.pt",
                     epsilon = 0.2,
                     epoch_num = 50,
                     lr_train = 0.005,
                     momentum = 0.1):
        """parse_params.

        Parameters
        ----------
        save_dir :
            dir
        save_model :
            Whether to save model
        save_name :
            model name
        epsilon :
            attack perturbation constraint
        epoch_num :
            number of training epoch
        lr_train :
            training learning rate
        momentum :
            momentum for optimizor
        """
        self.save_model = True
        self.save_dir = save_dir
        self.save_name = save_name
        self.epsilon = epsilon
        self.epoch_num = epoch_num
        self.lr_train = lr_train
        self.momentum = momentum

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

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)

            data_adv, output = self.adv_data(data, target, ep = self.epsilon)

            loss = self.calculate_loss(output, target)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            #print every 10
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(10*bs)))
                correct = 0


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
            data_adv, output_adv = self.adv_data(data, target, ep = self.epsilon)

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

    def adv_data(self, data, output, ep = 0.3, num_steps = 40):
        """Generate adversarial data for training.

        Parameters
        ----------
        data :
            data
        output :
            output
        ep :
            epsilon, perturbation budget.
        num_steps :
            iteration steps
        """
        # """
        # Generate input(adversarial) data for training.

        # """
        adversary = FGSM(self.model)
        data_adv = adversary.generate(data, output.flatten(), epsilon = ep)
        output = self.model(data_adv)

        return data_adv, output

    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss

