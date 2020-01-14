import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from DeepRobust.image.attack.Nattack import NATTACK
from DeepRobust.image.netmodels.CNN import Net


#initialize model
model = Net()
model.load_state_dict(torch.load("DeepRobust/image/save_models/MNIST_CNN_epoch_20.pt", map_location = torch.device('cpu')))
model.eval()
print("----------model_parameters-----------")

for names,parameters in model.named_parameters():
    print(names,',', parameters.type())
print("-------------------------------------")
data_loader = torch.utils.data.DataLoader(
              datasets.MNIST('DeepRobust/image/data', train = True,
              download = True,
              transform = transforms.Compose([transforms.ToTensor()])),
              batch_size = 1,
              shuffle = True)

attack = NATTACK(model)
attack.generate(dataloader = data_loader, classnum = 10)
