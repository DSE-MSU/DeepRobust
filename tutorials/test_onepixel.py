import torch
from torchvision import datasets, transforms

from deeprobust.image.attack.onepixel import Onepixel
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.CNN as CNN
from deeprobust.image.config import attack_params
import matplotlib.pyplot as plt

model = resnet.ResNet18().to('cuda')
print("Load network")

model.load_state_dict(torch.load("./trained_models/CIFAR10_ResNet18_epoch_50.pt"))
model.eval()

transform_val = transforms.Compose([
                transforms.ToTensor(),
                ])

test_loader  = torch.utils.data.DataLoader(
                datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                transform = transform_val),
                batch_size = 1, shuffle=True) #, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

xx, yy = next(iter(test_loader))
xx = xx.to('cuda').float()

onepixel_params = {
    'pixels':1
}
print(xx.size())
attack = Onepixel(model,'cuda')
success, rate = attack.generate(image = xx, label = yy, **onepixel_params)
print(success, rate)
