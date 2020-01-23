import torch
from torchvision import datasets,models,transforms
import torch.nn.functional as F 

import numpy as np
import argparse
import matplotlib.pyplot as plt

from DeepRobust.image.netmodels.CNN import Net
from DeepRobust.image import utils

def run_attack(attackmethod, batch_size, batch_num, device, test_loader, **kwargs):
    test_loss = 0
    correct = 0
    samplenum = 1000
    count = 0
    
    for count, (data, target) in enumerate(test_loader):
        if count > batch_num:
            break
        print('batch:{}'.format(count))

        data, target = data.to(device), target.to(device)
        adv_example = attackmethod.generate(data, target, **kwargs)
            
        output = model(adv_example)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability.

        #print(pred, target)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("===== ACCURACY =====")
    print('Attack Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_num * batch_size,
        100. * correct / (batch_num * batch_size)))



def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.")

    parser.add_argument("--attack_method", 
                        default = 'PGD', 
                        help = "choose a attack algorithm.")
    parser.add_argument("--attack_model", 
                        default = 'MNIST_CNN_epoch_20.pt',
                        help = "choose a attack model.")
    parser.add_argument("--dataset", 
                        default = 'MNIST',
                        help = "choose a dataset.")                    
    parser.add_argument("--epsilon", type = float, default = 0.3)
    parser.add_argument("--batch_num", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 1000)
    parser.add_argument("--num_steps", type = int, default = 40)
    parser.add_argument("--step_size", type = float, default = 0.01)
    parser.add_argument("--device", default = 'cuda',
                        help = "choose the device.")
    parser.add_argument("--path", 
                        default = "./defense_model/")

    return parser.parse_args()


if __name__ == "__main__":
    # read arguments
    args = parameter_parser() # read argument and creat an argparse object
    utils.tab_printer(args)

    model = Net()
    model.load_state_dict(torch.load(args.path + args.attack_model))
    model.eval()
    print("Loading network from" + args.path + args.attack_model)

    #load datasets
    if(args.dataset == "MNIST"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('DeepRobust/image/data', train = False,
                        download = True,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading MNIST dataset.")
    
    elif(args.dataset == "CIFAR" or args.dataset == 'CIFAR10'):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('DeepRobust/image/data', train = False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Loading CIFAR10 dataset.")

    elif(args.dataset == "ImageNet"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('DeepRobust/image/data', train=False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading ImageNet dataset.")


    print("===== START ATTACK =====")
    if(args.attack_method == "PGD"):
        from DeepRobust.image.attack.pgd import PGD
        attack_method = PGD(model, args.device)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon) 
    
    elif(args.attack_method == "FGSM"):
        from DeepRobust.image.attack.fgsm import FGM
        attack_method = FGM(model, args.device)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon) 
    
    elif(args.attack_method == "LBFGS"):
        pass
    
    elif(args.attack_method == "CW"):
        from DeepRobust.image.attack.cw import CarliniWagner
        attack_method = CarliniWagner(model, args.device)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader)
    
    elif(args.attack_method == "deepfool"):
        from DeepRobust.image.attack.deepfool import deepfool
        attack_method = DeepFool(model, args.device)
    
    elif(args.attack_method == "Nattack"):
        pass

