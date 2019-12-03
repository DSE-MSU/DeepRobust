import torch 
import numpy as np
from DeepRobust.image.defense.netmodels.CNNmodel import Net

LEVELS = 16

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    bs = train_loader.batch_size
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        Encoding = Thermometer(data, levels)

        output = model(Encoding)
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

def Thermometer(x, levels, flattened = False):

def one_hot_to_thermometer(x, levels, flattened = False):
    if flattened:
        pass
        #TODO: check how to flatten
    
    
    thermometer = torch.flip(torch.cumsum(x , dim = 1), [1])

    if flattened:
        pass
    return thermometer 

  