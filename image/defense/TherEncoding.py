import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from torchvision import datasets, transforms
from DeepRobust.image.netmodels.CNN import Net

import logging
import ipdb

LEVELS = 10

def train(model, device, train_loader, optimizer, epoch):
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

        print(encoding.size())
        output = model(encoding)
        print(output)
        loss = F.nll_loss(output, target)
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
        a = input()

def Thermometer(x, levels, flattened = False):
    """

    Output: Thermometer Encoding of the input.
    """

    onehot = one_hot(x, levels)

    thermometer = one_hot_to_thermometer(onehot, levels)

    return thermometer

def one_hot(x, levels):
    """
    Output: One hot Encoding of the input.
    """

    batch_size, channel, H, W = x.size()
    x = x.unsqueeze_(4)
    x = torch.ceil(x * (LEVELS-1)).long()
    onehot = torch.zeros(batch_size, channel, H, W, levels).float().to('cuda').scatter_(4, x, 1)
    print(onehot)

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
        datasets.MNIST('DeepRobust/image/data', train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=100,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('DeepRobust/image/data', train=False,
                    transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1000,
        shuffle=True)

    #ipdb.set_trace()

    print(train_loader)
    print(train_loader.data.size())
    channel = 1
    model = Net(in_channel1 = channel * LEVELS, out_channel2 = 30).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
    logger.info('Load model.')

    save_model = True
    for epoch in range(1, 100 + 1):     ## 5 batches
        print('Running epoch ', epoch)

        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        # if (save_model):
        #     torch.save(model.state_dict(), "../save_models/thermometer_encoding.pt")


