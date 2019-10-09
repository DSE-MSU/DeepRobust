from fgsm import FGM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
from fgsm import FGM
from CNNmodel import Net

model = Net()
print("Hello")
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

xx = datasets.MNIST('../data').data[8888]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()

## Set Targetå
yy = datasets.MNIST('../data', download=True).targets[8888]
yy = yy.unsqueeze_(0).float()

from PGD import LinfPGDAttack

adversary = LinfPGDAttack(model)

Adv_exp = adversary.perturb(xx, random_start = False)

print(Adv_exp)