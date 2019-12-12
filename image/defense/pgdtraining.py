import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image
from DeepRobust.image.attack.pgd import PGD
from DeepRobust.image.netmodels.CNN import Net

from DeepRobust.image.attack.pgd import PGD

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    bs = train_loader.batch_size

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        adversary = PGD(model)
        AdvExArray = adversary.generate(data, target, epsilon = 0.3, num_steps = 40)

        output = model(AdvExArray)
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        #print every 10
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(10*bs)))
            correct = 0


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # print clean accuracy
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # print adversarial accuracy
            adversary = PGD(model)
            data_adv = adversary.generate(data, pred, epsilon = 0.3, num_steps = 40)
            output_adv = model(data_adv)
            test_loss2 += F.nll_loss(output-adv, target, reduction='sum').item()  # sum up batch loss
            pred2 = output_adv.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct2 += pred.eq(target.view_as(pred2)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_loss2 /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    print('\nTest set: Average Adv loss: {:.3f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss2, correct2, len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))


if __name__ =='__main__':
    # model = CNNmodel.Net()
    # print("Load network")
    # model.load_state_dict(torch.load("../save_models/mnist_cnn.pt"))
    # model.eval()

    torch.manual_seed(100)
    device = torch.device("cuda")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('DeepRobust/image/defense/data', train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=100,
        shuffle=True)  ## han

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('DeepRobust/image/defense/data', train=False,
                    transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1000,
        shuffle=True)  ## han

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

    save_model = True
    for epoch in range(1, 100 + 1):     ## 5 batches
        print(epoch, flush = True)  ## han
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        if (save_model):
            torch.save(model.state_dict(), "DeepRobust/image/save_models/mnist_pgdtraining.pt")  ## han

