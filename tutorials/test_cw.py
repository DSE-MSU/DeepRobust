import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

import logging

from DeepRobust.image.attack.cw import CarliniWagner
from DeepRobust.image.netmodels.CNNmodel import Net

# print log
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Start test cw attack")

# load model 
model = Net()
model.load_state_dict(torch.load("DeepRobust/image/save_models/mnist_cnn.pt", map_location = torch.device('cuda')))
model.eval()

xx = datasets.MNIST('DeepRobust/image/data', download = False).data[8888]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float().to('cuda')

## Set Targetå
yy = datasets.MNIST('DeepRobust/image/data', download = False).targets[8888]
yy = yy.float()

cw_params = {
    'confidence': 1e-4,
    'clip_max': 1,
    'clip_min': 0,
    'max_iterations': 1000,
    'initial_const': 1e-2,
    'binary_search_steps': 5,
    'learning_rate': 5e-3,
    'abort_early': True,
}

attack = CarliniWagner(model, device='cuda')
AdvExArray = attack.generate(xx, yy, target = 1, classnum = 10, **cw_params)
Adv = AdvExArray.clone()

# test the result
predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

# AdvExArray = torch.from_numpy(AdvExArray)
predict1 = model(Adv)
predict1= predict1.argmax(dim=1, keepdim=True)

print(predict0)
print(predict1)

import matplotlib.pyplot as plt
Adv = Adv.cpu()
plt.imshow(Adv[0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample_cw.png')

