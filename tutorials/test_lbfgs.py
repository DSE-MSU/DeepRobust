from lbfgs import LBFGS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
from fgsm import FGM
from DeepRobust.image.netmodels.CNNmodel import Net


#load model 
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location = torch.device('cpu')))
model.eval()

xx = datasets.MNIST('../data', download = True).data[8888]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()

## Set Targetå
yy = datasets.MNIST('../data').targets[8888]
yy = yy.float()

lbfgs_params = {
    'epsilon': 1e-4,
    'maxiter': 20,
    'clip_max': 1,
    'clip_min': 0,
    'class_num': 10
}

attack = LBFGS(model, 8, device='cpu')
AdvExArray, dis, loss, valueofc= attack.generate(xx, yy, **lbfgs_params)
AdvExArray = AdvExArray.unsqueeze_(0).float()

#test the result
predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

#AdvExArray = torch.from_numpy(AdvExArray)
predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1, 'distance:',dis, 'target func:',loss, 'c:', valueofc)

import matplotlib.pyplot as plt

plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample.png')



