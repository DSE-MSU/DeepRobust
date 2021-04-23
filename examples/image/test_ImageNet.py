import torchvision
import torch
from torchvision import datasets
from torchvision import transforms
import os
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params

val_root  = '/mnt/home/liyaxin1/Documents/data/ImageNet'
#Imagenet_data = torchvision.datasets.ImageNet(val_root, split = 'val')
test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('~/Documents/data/ImageNet/val', transforms.Compose([
                    transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=1, shuffle=False)

#import torchvision.models as models
#model = models.resnet50(pretrained=True).to('cuda')

import pretrainedmodels
model = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').to('cuda')

for i, (input, y) in enumerate(test_loader):

    import ipdb
    ipdb.set_trace()

    input, y = input.to('cuda'), y.to('cuda')
    pred = model(input)
    print(pred.argmax(dim=1, keepdim = True))

    adversary = PGD(model)
    AdvExArray = adversary.generate(input, y, **attack_params['PGD_CIFAR10']).float()


