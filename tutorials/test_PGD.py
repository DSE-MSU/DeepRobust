import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from DeepRobust.image.attack.pgd import PGD
import DeepRobust.image.netmodels.resnet as resnet
import DeepRobust.image.netmodels.CNNmodel as CNNmodel
import matplotlib.pyplot as plt


model = resnet.ResNet18().to('cuda')
print("Load network")
model.load_state_dict(torch.load("DeepRobust/image/save_models/CIFAR10_ResNet18_epoch_50.pt"))
model.eval()

transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10('DeepRobust/image/data', train = False, download=True,
                transform = transform_val),
                batch_size = 1, shuffle=True) #, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

xx, yy = next(iter(test_loader))
xx = xx.to('cuda')

plt.imshow(xx.cpu().numpy().swapaxes(1,3).swapaxes(1,2)[0], vmin = 0, vmax = 255)
plt.savefig('DeepRobust/image/adversary_examples/cifar_advexample_orig.png')

xx = xx.to('cuda').float()
print('true label',classes[yy])

predict0 = model(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model)
AdvExArray = adversary.generate(xx,yy)

predict1 = model(AdvExArray)
predict1= predict1.argmax(dim=1, keepdim=True)

print('predict_orig',classes[predict0])
print('predict_adv',classes[predict1])

AdvExArray = AdvExArray.cpu().detach().numpy()

AdvExArray = AdvExArray.swapaxes(1,3)
AdvExArray = AdvExArray.swapaxes(1,2)
plt.imshow(AdvExArray[0], vmin = 0, vmax = 255)
plt.savefig('DeepRobust/image/adversary_examples/cifar_advexample_pgd.png')


        