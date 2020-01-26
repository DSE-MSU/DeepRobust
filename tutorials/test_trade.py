import torch
from torchvision import datasets, transforms
import numpy as np

from DeepRobust.image.defense.trades import TRADES
from DeepRobust.image.netmodels.CNN import Net

train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('DeepRobust/image/defense/data', train = True, download = True,
                transform = transforms.Compose([transforms.ToTensor()])),
                batch_size = 100,
                shuffle = True)  

test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('DeepRobust/image/defense/data', train = False,
            transform = transforms.Compose([transforms.ToTensor()])),
            batch_size = 1000,
            shuffle = True)  


model = Net()
defense = TRADES(model,'cuda')
defense.generate(train_loader, test_loader)



