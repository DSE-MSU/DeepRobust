import torch
from torchvision import datasets, transforms
import numpy as np

from DeepRobust.image.defense.trades import TRADES
from DeepRobust.image.netmodels.CNN import Net

train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('DeepRobust/image/defense/data', train = True, download = True,
                transform = transforms.Compose([transforms.ToTensor()])),
                batch_size = 100,
                shuffle = True)  

test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('DeepRobust/image/defense/data', train = False,
            transform = transforms.Compose([transforms.ToTensor()])),
            batch_size = 1000,
            shuffle = True)  


model = Net()
defense = TRADES(model,'cuda')
defense.generate(train_loader, test_loader)

xx = datasets.MNIST('../data').data[3333]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()

## Set Targetå
yy = datasets.MNIST('DeepRobust/image/data', download=True).targets[3333]
yy = yy.unsqueeze_(0).float()

predict0 = defensemodel(xx)
predict0= predict0.argmax(dim=1, keepdim=True)

adversary = PGD(model)
AdvExArray = adversary.generate(xx,yy, epsilon = 0.3)

# predict1 = model(AdvExArray)
# predict1= predict1.argmax(dim=1, keepdim=True)

AdvExArray = AdvExArray.cpu()
predict2 = defensemodel(AdvExArray)
predict2= predict2.argmax(dim=1, keepdim=True)

print(predict0)
# print(predict1)
print(predict2)

AdvExArray = AdvExArray.cpu().detach().numpy()

