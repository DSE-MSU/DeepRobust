import torch
from torchvision import datasets, transforms

from DeepRobust.image.attack.onepixel import Onepixel
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

onepixel_params = {
    'pixels':1
}
print(xx.size())
attack = Onepixel(model,'cuda')
AdvImg = attack.generate(image = xx, label = yy, **onepixel_params)
AdvImg = AdvImg.cpu().detach().numpy()
AdvImg = AdvImg.swapaxes(1,3).swapaxes(1,2)[0]


import matplotlib.pyplot as plt
plt.imshow(AdvImg,cmap='gray',vmin=0,vmax=255)
plt.savefig('advexample_onepixel.png')
