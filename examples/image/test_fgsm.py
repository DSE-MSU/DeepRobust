import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
import argparse

from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.")

    parser.add_argument("--destination",
                        default = './trained_models/',
                        help = "choose destination to load the pretrained models.")

    parser.add_argument("--filename",
                        default = "MNIST_CNN_epoch_20.pt")

    return parser.parse_args()

args = parameter_parser() # read argument and creat an argparse object

model = Net()

model.load_state_dict(torch.load(args.destination + args.filename))
model.eval()
print("Finish loading network.")

xx = datasets.MNIST('./', download = False).data[999:1000].to('cuda')
xx = xx.unsqueeze_(1).float()/255
#print(xx.size())

## Set Target
yy = datasets.MNIST('./', download = False).targets[999:1000].to('cuda')

"""
Generate adversarial examples
"""

F1 = FGSM(model, device = "cuda")       ### or cuda
AdvExArray = F1.generate(xx, yy, **attack_params['FGSM_MNIST'])


predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print("original prediction:")
print(predict0)

print("attack prediction:")
print(predict1)

xx = xx.cpu().detach().numpy()
AdvExArray = AdvExArray.cpu().detach().numpy()

import matplotlib.pyplot as plt
plt.imshow(xx[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('./adversary_examples/mnist_advexample_fgsm_ori.png')

plt.imshow(AdvExArray[0,0]*255,cmap='gray',vmin=0,vmax=255)
plt.savefig('./adversary_examples/mnist_advexample_fgsm_adv.png')
