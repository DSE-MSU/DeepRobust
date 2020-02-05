from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import CNNmodel 

torch.manual_seed(100)
device = torch.device("cuda")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                 transform=transforms.Compose([transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=1000,
    shuffle=True)

model = CNNmodel.Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

save_model = True
for epoch in range(1, 5 + 1):     ## 5 batches
    print(epoch)
    CNNmodel.train(model, device, train_loader, optimizer, epoch)
    CNNmodel.test(model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
