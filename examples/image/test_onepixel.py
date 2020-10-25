import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image
import argparse

from deeprobust.image.attack.onepixel import Onepixel
from deeprobust.image.netmodels import resnet
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

model = resnet.ResNet18().to('cuda')
print("Load network")

model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_20.pt"))
model.eval()

transform_val = transforms.Compose([
                transforms.ToTensor(),
                ])

test_loader  = torch.utils.data.DataLoader(
                datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                transform = transform_val),
                batch_size = 1, shuffle=True) #, **kwargs)


classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

xx, yy = next(iter(test_loader))
xx = xx.to('cuda').float()

"""
Generate adversarial examples
"""

F1 = Onepixel(model, device = "cuda")       ### or cuda
AdvExArray = F1.generate(xx, yy)

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
xx = xx[0].transpose(1, 2, 0)
AdvExArray = AdvExArray[0].transpose(1, 2, 0)

plt.imshow(xx, vmin=0, vmax=255)
plt.savefig('./adversary_examples/cifar10_advexample_ori.png')

plt.imshow(AdvExArray, vmin=0, vmax=255)
plt.savefig('./adversary_examples/cifar10_advexample_onepixel_adv.png')
