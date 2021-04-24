import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from lbfgs import LBFGS
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params

#load model
model = Net()
model.load_state_dict(torch.load("/home/bizon/Desktop/liyaxin/deeprobust_trained_model/MNIST_CNN_epoch_20.pt", map_location = torch.device('cpu')))
model.eval()

import ipdb
ipdb.set_trace()

xx = datasets.MNIST('deeprobust/image/data', download = True).data[8888]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()

## Set Targetå
yy = datasets.MNIST('deeprobust/image/data', download = False).targets[8888]
yy = yy.float()

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

attack_param = {
    'epsilon': 2e-1,
    'maxiter': 20,
    'clip_max': 1,
    'clip_min': 0,
    'class_num': 10
    }

attack = LBFGS(model, device='cpu')
AdvExArray = attack.generate(xx, yy, target_label = 2, **attack_param)

#AdvExArray = torch.from_numpy(AdvExArray)
predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)

import matplotlib.pyplot as plt

plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample.png')



