import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
from attack import fgsm
from DeepRobust.image.netmodels.CNNmodel import Net

model = CNNmodel.Net()
print("Load network")
model.load_state_dict(torch.load("save_models/mnist_cnn.pt"))
model.eval()

xx = datasets.MNIST('../data').data[999:1000].to('cuda')
xx = xx.unsqueeze_(1).float()/255

## Set Targetå
yy = datasets.MNIST('../data', download=True).targets[999:1000].to('cuda')

fgsm_params = {
    'epsilon': 0.2,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
}

F1 = fgsm.FGM(model, device = "cuda")       ### or cuda
AdvExArray = F1.generate(xx, yy, **fgsm_params)

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
