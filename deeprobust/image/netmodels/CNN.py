"""
This is an implementatio of a Convolution Neural Network with 2 Convolutional layer.
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

class Net(nn.Module):
    """Model counterparts.
    """

    def __init__(self, in_channel1 = 1, out_channel1 = 32, out_channel2 = 64, H = 28, W = 28):
        super(Net, self).__init__()
        self.H = H
        self.W = W
        self.out_channel2 = out_channel2

        ## define two convolutional layers
        self.conv1 = nn.Conv2d(in_channels = in_channel1,
                               out_channels = out_channel1,
                               kernel_size = 5,
                               stride= 1,
                               padding = (2,2))
        self.conv2 = nn.Conv2d(in_channels = out_channel1,
                               out_channels = out_channel2,
                               kernel_size = 5,
                               stride = 1,
                               padding = (2,2))

        ## define two linear layers
        self.fc1 = nn.Linear(int(H/4)*int(W/4)* out_channel2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, int(self.H/4) * int(self.W/4) * self.out_channel2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, int(self.H/4) * int(self.W/4) * self.out_channel2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    """train network.

    Parameters
    ----------
    model :
        model
    device :
        device(option:'cpu','cuda')
    train_loader :
        training data loader
    optimizer :
        optimizer
    epoch :
        epoch
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        #print every 10
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """test network.

    Parameters
    ----------
    model :
        model
    device :
        device(option:'cpu', 'cuda')
    test_loader :
        testing data loader
    """
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



