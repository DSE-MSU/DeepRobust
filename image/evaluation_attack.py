import torch
from torchvision import datasets,models,transforms
import torch.nn.functional as F 

import numpy as np
import argparse
import matplotlib.pyplot as plt

import attack
from netmodels.CNNmodel import Net

def run_attack(attackmethod, batch_size, batch_num, device, test_loader, **kwargs):
    test_loss = 0
    correct = 0
    samplenum = 1000
    count = 0
    
    with torch.no_grad():
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

    print('Attack Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_num * batch_size,
        100. * correct / (batch_num * batch_size)))



def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.")

    parser.add_argument("--attack_method", 
                        default = 'PGD_attack', 
                        help = "choose a attack algorithm.")
    parser.add_argument("--attack_model", 
                        default = 'mnist_cnn.pt',
                        help = "choose a attack model.")
    parser.add_argument("--epsilon", type = float, default = 0.3)
    parser.add_argument("--batch_num", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 1000)
    parser.add_argument("--num_steps", type = int, default = 40)
    parser.add_argument("--step_size", type = float, default = 0.01)
    parser.add_argument("--device",  default = 'cuda',
                        help = "choose the device.")

    return parser.parse_args()


if __name__ == "__main__":
    # read arguments
    args = parameter_parser() # read argument and creat an argparse object
    model = Net()

    model.load_state_dict(torch.load("./save_models/"+args.attack_model))
    model.eval()
    print("Load network.")

    #load datasets
    if(args.attack_method == "MNIST"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('../data', train=False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Load MNIST Dataset")
    
    elif(args.attack_method == "CIFAR"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('../data', train=False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Load CIFAR10 Dataset")

    elif(args.attack_method == "ImageNet"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('../data', train=False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Load ImageNet Dataset")      

    
    if(args.attack_method == "PGD_attack"):
        attack_method = attack.pgd.PGD(model)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon) 
    elif(args.attack_method == "FGSM_attack"):
        attack_method = attack.fgsm.FGM(model)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon) 

