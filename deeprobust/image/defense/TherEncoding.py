"""
This is an implementation of Thermometer Encoding.

References
----------
.. [1] Buckman, Jacob, Aurko Roy, Colin Raffel, and Ian Goodfellow. "Thermometer encoding: One hot way to resist adversarial examples." In International Conference on Learning Representations. 2018.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from torchvision import datasets, transforms
from deeprobust.image.netmodels.CNN import Net

import logging

## TODO
# class ther_attack(pgd_attack):
#     """
#     PGD attacks in response to thermometer encoding models
#     """
## TODO
# def adv_train():
#     """
#     adversarial training for thermomoter encoding
#     """

def train(model, device, train_loader, optimizer, epoch):
    """training process.

    Parameters
    ----------
    model :
        model
    device :
        device
    train_loader :
        training data loader
    optimizer :
        optimizer
    epoch :
        epoch
    """
    logger.info('trainging')
    model.train()
    correct = 0
    bs = train_loader.batch_size

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        encoding = Thermometer(data, LEVELS)
        encoding = encoding.permute(0, 2, 3, 1, 4)
        encoding = torch.flatten(encoding, start_dim = 3)
        encoding = encoding.permute(0, 3, 1, 2)

        #print(encoding.size())

        #ipdb.set_trace()
        output = model(encoding)

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        #print(pred,target)
        #print every 10
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(10*bs)))
            correct = 0
        a = input()


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            encoding = Thermometer(data, LEVELS)
            encoding = encoding.permute(0, 2, 3, 1, 4)
            encoding = torch.flatten(encoding, start_dim=3)
            encoding = encoding.permute(0, 3, 1, 2)

            # print clean accuracy
            output = model(encoding)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Clean loss: {:.3f}, Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def Thermometer(x, levels, flattened = False):
    """
    Output
    ------
    Thermometer Encoding of the input.
    """

    onehot = one_hot(x, levels)

    thermometer = one_hot_to_thermometer(onehot, levels)

    return thermometer

def one_hot(x, levels):
    """
    Output
    ------
    One hot Encoding of the input.
    """

    batch_size, channel, H, W = x.size()
    x = x.unsqueeze_(4)
    x = torch.ceil(x * (LEVELS-1)).long()
    onehot = torch.zeros(batch_size, channel, H, W, levels).float().to('cuda').scatter_(4, x, 1)
    #print(onehot)

    return onehot

def one_hot_to_thermometer(x, levels, flattened = False):
    """
    Convert One hot Encoding to Thermometer Encoding.
    """

    if flattened:
        pass
        #TODO: check how to flatten

    thermometer = torch.cumsum(x , dim = 4)

    if flattened:
        pass
    return thermometer

if __name__ =='__main__':

    logger = logging.getLogger('Thermometer Encoding')

    handler = logging.StreamHandler()  # Handler for the logger
    handler.setFormatter(logging.Formatter('%(asctime)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info('Start attack.')

    torch.manual_seed(100)
    device = torch.device("cuda")

    #ipdb.set_trace()

    logger.info('Load trainset.')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('deeprobust/image/data', train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=100,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('deeprobust/image/data', train=False,
                    transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1000,
        shuffle=True)

    #ipdb.set_trace()

    #TODO: change the channel according to the dataset.
    LEVELS = 10
    channel = 1
    model = Net(in_channel1 = channel * LEVELS, out_channel1= 32 * LEVELS, out_channel2= 64 * LEVELS).to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.2)
    logger.info('Load model.')

    save_model = True
    for epoch in range(1, 50 + 1):     ## 5 batches
        print('Running epoch ', epoch)

        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        if (save_model):
            torch.save(model.state_dict(), "deeprobust/image/save_models/thermometer_encoding.pt")


