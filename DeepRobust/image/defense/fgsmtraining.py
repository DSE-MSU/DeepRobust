import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

from DeepRobust.image.netmodels import CNN
from DeepRobust.image.attack.fgsm import FGM
from DeepRobust.image.defense.base_defense import BaseDefense

class FGSMtraining(BaseDefense):
    def __init__(self, model, device):

        self.device = device
        self.model = model

    def generate(self, train_loader, test_loader, **kwargs):
        """
        FGSM defense process:
        """
        self.parse_params(**kwargs)
        
        torch.manual_seed(100)
        device = torch.device(self.device)

        optimizer = optim.Adam(self.model.parameters(), self.lr, momentum=0.1)
    
        save_model = True
        for epoch in range(1, 100 + 1):    
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
                     save_name = "mnist_fgsmtraining_0.3.pt",
                     epsilon = 0.2,
                     num_steps = 40,
                     lr = 0.001,
                     momentum = 0.1):
        # """
        # Set parameters for pgd training.
        # """
        self.save_model = True
        self.save_dir = save_dir
        self.save_name = save_name
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum

    def train(self, device, train_loader, optimizer, epoch):
        """
        Training process.
        """
        self.model.train()
        correct = 0
        bs = train_loader.batch_size

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            
            data, target = data.to(device), target.to(device)

            data_adv, output = self.adv_data(data, target, ep = self.epsilon, num_steps = self.num_steps)

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
        Testing process.
    
        """
        model.eval()

        test_loss = 0
        correct = 0
        test_loss_adv = 0
        correct_adv = 0
        with torch.no_grad():
             for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # print clean accuracy
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
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
            
    def adv_data(self, data, output, ep = 0.3, num_steps = 40):
        # """
        # Generate input(adversarial) data for training.
        
        # """
        adversary = FGM(self.model)
        data_adv = adversary.generate(data, output.flatten(), epsilon = ep, num_steps = num_steps)
        output = self.model(data_adv)

        return data_adv, output

    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """
        
        loss = F.nll_loss(output, target, reduction = redmode)
        return loss
    