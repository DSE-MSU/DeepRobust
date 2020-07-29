import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchvision import models, datasets

from deeprobust.image.attack.deepfool import DeepFool
import deeprobust.image.netmodels.resnet as resnet
import matplotlib.pyplot as plt

'''
CIFAR10
'''

# load model
model = resnet.ResNet18().to('cuda')
print("Load network")

"""
Change the model directory here
"""
model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_20.pt"))
model.eval()

# load dataset
testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('image/data', train = False, download = True,
    transform = transforms.Compose([transforms.ToTensor()])),
    batch_size = 1, shuffle = True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# choose attack example
X, Y = next(iter(testloader))
X = X.to('cuda').float()

# run deepfool attack
adversary = DeepFool(model)
AdvExArray = adversary.generate(X, Y).float()

# predict
pred = model(AdvExArray).cpu().detach()

# print and save result
print('===== RESULT =====')
print("true label:", classes[Y])
print("predict_adv:", classes[np.argmax(pred)])

AdvExArray = AdvExArray.cpu().detach().numpy()
AdvExArray = AdvExArray.swapaxes(1,3).swapaxes(1,2)[0]

plt.imshow(AdvExArray, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_deepfool.png')

