import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.netmodels.CNN import Net


#initialize model
model = Net()
model.load_state_dict(torch.load("trained_models/mnist_fgsmtraining_0.2.pt", map_location = torch.device('cuda')))
model.eval()
print("----------model_parameters-----------")

for names,parameters in model.named_parameters():
    print(names,',', parameters.type())
print("-------------------------------------")
data_loader = torch.utils.data.DataLoader(
              datasets.MNIST('deeprobust/image/data', train = True,
              download = True,
              transform = transforms.Compose([transforms.ToTensor()])),
              batch_size = 1,
              shuffle = True)

attack = NATTACK(model)
attack.generate(dataloader = data_loader, classnum = 10)
