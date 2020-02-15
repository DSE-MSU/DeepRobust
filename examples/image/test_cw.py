import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

import logging

from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params

# print log
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Start test cw attack")

# load model
model = Net()
model.load_state_dict(torch.load("./trained_models/MNIST_CNN_epoch_20.pt", map_location = torch.device('cuda')))
model.eval()

xx = datasets.MNIST('deeprobust/image/data', download = False).data[1234]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float().to('cuda')

## Set Targetå
yy = datasets.MNIST('deeprobust/image/data', download = False).targets[1234]
yy = yy.float()


attack = CarliniWagner(model, device='cuda')
AdvExArray = attack.generate(xx, yy, target_label = 1, classnum = 10, **attack_params['CW_MNIST'])
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
plt.imshow(Adv[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('./adversary_examples/mnist_advexample_cw.png')

