import attack
from CNNmodel import Net
import torch
from torchvision import datasets,models,transforms


model = Net()
print("Hello")
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

## Set Targetå
yy = datasets.MNIST('../data', download=True).targets[1003]
yy = yy.unsqueeze_(0).float()

att = attack.PGD(model)
xx = datasets.MNIST('../data').data[1003]
xx = xx.unsqueeze_(0).float()/255
xx = xx.unsqueeze_(0).float()
    
import matplotlib.pyplot as plt
plt.imshow(AdvExArray[0,0]*255, cmap='gray', vmin = 0, vmax = 255)
plt.savefig('advexample_pgd.png')
print("adv_save")