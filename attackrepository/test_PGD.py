from fgsm import FGM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from pgd import PGD
from CNNmodel import Net

model = Net()
print("Hello")
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

xx = datasets.MNIST('../data').data[999]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()

## Set Targetå
yy = datasets.MNIST('../data', download=True).targets[999]
yy = yy.unsqueeze_(0).float()

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model)
AdvExArray = adversary.generate(xx,yy)

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)

AdvExArray = AdvExArray.cpu().detach().numpy()

import matplotlib.pyplot as plt
plt.imshow(AdvExArray[0,0]*255, cmap='gray', vmin = 0, vmax = 255)
plt.savefig('advexample_pgd.png')