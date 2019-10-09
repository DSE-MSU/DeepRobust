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

fgsm_params = {
    'epsilon': 0.2,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
}

F1 = FGM(model, device = "cuda")       ### or cuda
AdvExArray = F1.generate(image=xx, label=yy, **fgsm_params)

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

AdvExArray = torch.from_numpy(AdvExArray)
predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)

import matplotlib.pyplot as plt
plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample.png')
