import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

import sys
sys.path.append("..")
from netmodels import CNNmodel
import attack

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)
        adversary = attack.fgsm.FGM(model)
        AdvExArray = adversary.generate(data, target)

        output = model(AdvExArray)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        #print every 10 
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ =='__main__':
    # model = CNNmodel.Net()
    # print("Load network")
    # model.load_state_dict(torch.load("../save_models/mnist_cnn.pt"))
    # model.eval()
    
    torch.manual_seed(100)
    device = torch.device("cuda")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=64,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1000,
        shuffle=True)

    model = CNNmodel.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

    save_model = True
    for epoch in range(1, 100 + 1):     ## 5 batches
        #print('1',epoch)
        print(epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        if (save_model):
            torch.save(model.state_dict(), "../save_models/mnist_fgsmtraining.pt")

