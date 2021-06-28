"""
This implementation is used to create adversarial dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from deeprobust.image.attack.pgd import PGD
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.CNN as CNN
from deeprobust.image.config import attack_params
import matplotlib.pyplot as plt
from deeprobust.image.config import attack_params

def main(args):
    #Load Model.
    model = resnet.ResNet18().to('cuda')
    print("Load network")

    model.load_state_dict(torch.load("~/Documents/deeprobust_model/cifar_res18_120.pt"))
    model.eval()

    transform_val = transforms.Compose([
                    transforms.ToTensor(),
                    ])
    train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor()])),
                    batch_size=128,
                    shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                    transform = transform_val),
                    batch_size = 128, shuffle=True) #, **kwargs)


    normal_data, adv_data = None, None
    adversary = PGD(model)

    for x, y in train_loader:
        x, y = x.cuda(), t.cuda()
        y_pred = model(x)
        train_acc += accuracy(y_pred, y)
        x_adv = adversary.generate(x, y, **attack_params['PGD_CIFAR10']).float()
        y_adv = model(x_adv)
        adv_acc += accuracy(y_adv, y)
        train_n += y.size(0)

        x, x_adv = x.data, x_adv.data
        if normal_data is None:
            normal_data, adv_data = x, x_adv
        else:
            normal_data = torch.cat((normal_data, x))
            adv_data = torch.cat((adv_data, x_adv))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(train_acc / train_n * 100, adv_acc / train_n * 100))
    torch.save({"normal": normal_data, "adv": adv_data}, "data.tar")
    torch.save({"state_dict": model.state_dict()}, "cnn.tar")