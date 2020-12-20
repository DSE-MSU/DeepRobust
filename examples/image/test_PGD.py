import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from deeprobust.image.attack.pgd import PGD
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.CNN as CNN
from deeprobust.image.config import attack_params
import matplotlib.pyplot as plt

model = resnet.ResNet18().to('cuda')
print("Load network")

import ipdb
ipdb.set_trace()

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

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model)
AdvExArray = adversary.generate(xx, yy, **attack_params['PGD_CIFAR10']).float()

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print('====== RESULT =====')
print('true label',classes[yy.cpu()])
print('predict_orig',classes[predict0.cpu()])
print('predict_adv',classes[predict1.cpu()])

x_show = xx.cpu().numpy().swapaxes(1,3).swapaxes(1,2)[0]
# print('xx:', x_show)
plt.imshow(x_show, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_orig.png')
# print('x_show', x_show)


# print('---------------------')
AdvExArray = AdvExArray.cpu().detach().numpy()
AdvExArray = AdvExArray.swapaxes(1,3).swapaxes(1,2)[0]

# print('Adv', AdvExArray)

# print('----------------------')
# print(AdvExArray)
plt.imshow(AdvExArray, vmin = 0, vmax = 255)
plt.savefig('./adversary_examples/cifar_advexample_pgd.png')


