from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def train(model, data, device, maxepoch, data_path, seed = 100):

    torch.manual_seed(seed)

    train_loader, test_loader = feed_dataset(data, data_path)

    if (model == 'CNN'):
        import deeprobust.image.netmodels.CNN as MODEL
        #from deeprobust.image.netmodels.CNN import Net
        train_net = MODEL.Net().to(device)

    elif (model == 'ResNet18'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet18().to(device)

    elif (model == 'ResNet34'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet34().to(device)

    elif (model == 'ResNet50'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet50().to(device)

    optimizer = optim.SGD(train_net.parameters(), lr=0.01, momentum=0.5)

    save_model = True
    for epoch in range(1, maxepoch + 1):     ## 5 batches

        print(epoch)
        MODEL.train(train_net, device, train_loader, optimizer, epoch)
        MODEL.test(train_net, device, test_loader)

        if (save_model and epoch %10 == 0):
            if os.path.isdir('./trained_models/'):
                print('Save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")
            else:
                os.mkdir('./trained_models/')
                print('Make directory and save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")

def feed_dataset(data, data_dict):
    if(data == 'CIFAR10'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=True, download = True,
                        transform=transform_train),
                 batch_size= 1000, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)

    elif(data == 'MNIST'):
        train_loader = torch.utils.data.DataLoader(
                 datasets.MNIST(data_dict, train=True, download = True,
                 transform=transforms.Compose([transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])),
                 batch_size=64,
                 shuffle=True)

        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, download = True,
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=1000,
                shuffle=True)

    elif(data == 'ImageNet'):
        pass

    return train_loader, test_loader



