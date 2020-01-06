import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from DeepRobust.image.attack.pgd import PGD
import DeepRobust.image.netmodels.resnet as resnet
import DeepRobust.image.netmodels.CNN as CNN
from DeepRobust.image.config import attack_params
import matplotlib.pyplot as plt

import ipdb

model = resnet.ResNet18().to('cuda')
print("Load network")

model.load_state_dict(torch.load("DeepRobust/image/save_models/CIFAR10_ResNet18_epoch_50.pt").state_dict())
model.eval()

transform_val = transforms.Compose([
                transforms.ToTensor(),
                ])

test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10('DeepRobust/image/data', train = False, download=True,
                transform = transform_val),
                batch_size = 1, shuffle=True) #, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

xx, yy = next(iter(test_loader))
xx = xx.to('cuda').float()
#ipdb.set_trace()

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model)
AdvExArray = adversary.generate(xx, yy, **attack_params['PGD_CIFAR10']).float()

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print('----------------------')
print('true label',classes[yy])
print('predict_orig',classes[predict0])
print('predict_adv',classes[predict1])

# mean = np.array([0.2023000, 0.1994000, 0.2012000])
# std = np.array([0.4914000, 0.4822000, 0.4465000])
x_show = xx.cpu().numpy().swapaxes(1,3).swapaxes(1,2)[0]
# # print('xx:', x_show)
# x_show = x_show * mean + std
plt.imshow(x_show, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_orig.png')
# print('x_show', x_show)


# print('---------------------')
AdvExArray = AdvExArray.cpu().detach().numpy()
AdvExArray = AdvExArray.swapaxes(1,3).swapaxes(1,2)[0]

# print('Adv', AdvExArray)


#AdvExArray = AdvExArray * mean + std
# print('----------------------')
# print(AdvExArray)
plt.imshow(AdvExArray, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_pgd.png')



