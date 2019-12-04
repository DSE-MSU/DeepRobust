import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np
from torchvision import datasets, transforms
from DeepRobust.image.netmodels.CNNmodel import Net

LEVELS = 10

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    bs = train_loader.batch_size
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device) 
        encoding = Thermometer(data, LEVELS)
        encoding = np.swapaxes(encoding.cpu(), 1, 3)
        encoding = torch.flatten(encoding, start_dim = 3)
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

    onehot = one_hot(x, levels)

    thermometer = one_hot_to_thermometer(onehot, levels)
    
    return thermometer

def one_hot(x, levels):

    batch_size, channel, H, W = x.size()
    x = x.unsqueeze_(4)
    x = torch.ceil(x * (LEVELS-1)).long()
    onehot = torch.zeros(batch_size, channel, H, W, levels).float().to('cuda').scatter_(4, x, 1)
    print(onehot)
    
    return onehot

def one_hot_to_thermometer(x, levels, flattened = False):
    if flattened:
        pass
        #TODO: check how to flatten
    
    thermometer = torch.cumsum(x , dim = 4)

    if flattened:
        pass
    return thermometer 

if __name__ =='__main__':    
    torch.manual_seed(100)
    device = torch.device("cuda")

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

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

    save_model = True
    for epoch in range(1, 100 + 1):     ## 5 batches
        print(epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        # if (save_model):
        #     torch.save(model.state_dict(), "../save_models/thermometer_encoding.pt")

  