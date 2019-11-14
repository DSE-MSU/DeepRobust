import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from attack import pgd
from DeepRobust.image.netmodels.CNNmodel import Net

model = Net()
print("Load network")
model.load_state_dict(torch.load("./save_models/mnist_cnn.pt"))
model.eval()

xx = datasets.MNIST('../data',train = False).data[1000:1001]
xx = xx.unsqueeze_(1).float()/255
print(xx)

## Set Targetå
yy = datasets.MNIST('../data', train = False, download=True).targets[1000:1001].float()
print(yy)

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = pgd.PGD(model)
AdvExArray = adversary.generate(xx,yy)

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)
print(predict0.cpu().eq(predict1.cpu().view_as(predict0)).sum().item())

AdvExArray = AdvExArray.cpu().detach().numpy()

import matplotlib.pyplot as plt
plt.imshow(AdvExArray[0,0]*255, cmap='gray', vmin = 0, vmax = 255)
plt.savefig('advexample_pgd.png')


        