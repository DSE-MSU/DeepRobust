import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from DeepRobust.image.attack.fgsm import FGM
from DeepRobust.image.netmodels.CNN import Net
from DeepRobust.image.config import attack_params

import ipdb

model = Net()

print("Load network")
model.load_state_dict(torch.load("DeepRobust/image/save_models/MNIST_CNN_epoch_10.pt"))
model.eval()

xx = datasets.MNIST('DeepRobust/image/data', download = False).data[999:1000].to('cuda')
xx = xx.unsqueeze_(1).float()/255
print(xx.size())

## Set Targetå
yy = datasets.MNIST('DeepRobust/image/data', download = False).targets[999:1000].to('cuda')


F1 = FGM(model, device = "cuda")       ### or cuda
AdvExArray = F1.generate(xx, yy, **attack_params['FGSM_MNIST'])

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)

AdvExArray = AdvExArray.cpu().detach().numpy()

import matplotlib.pyplot as plt
plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample_fgsm.png')
