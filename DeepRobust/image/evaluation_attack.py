import torch
from torchvision import datasets,models,transforms
import torch.nn.functional as F

import numpy as np
import argparse
import matplotlib.pyplot as plt
import random

from DeepRobust.image import utils

def run_attack(attackmethod, batch_size, batch_num, device, test_loader, random_targeted = False, **kwargs):
    test_loss = 0
    correct = 0
    samplenum = 1000
    count = 0
    classnum = 10
    for count, (data, target) in enumerate(test_loader):
        if count == batch_num:
            break
        print('batch:{}'.format(count))

        data, target = data.to(device), target.to(device)
        if(random_targeted == True):
            r = list(range(0, target)) + list(range(target+1, classnum))
            target_label = random.choice(r)
            adv_example = attackmethod.generate(data, target, target_label = target_label, **kwargs)

        else:
            adv_example = attackmethod.generate(data, target, **kwargs)

        output = model(adv_example)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability.

        correct += pred.eq(target.view_as(pred)).sum().item()

    batch_num = count+1
    test_loss /= len(test_loader.dataset)
    print("===== ACCURACY =====")
    print('Attack Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_num * batch_size,
        100. * correct / (batch_num * batch_size)))

def load_net(attack_model, filename, path):
    if(attack_model == "CNN"):
        from DeepRobust.image.netmodels.CNN import Net

    model = Net()
    model.load_state_dict(torch.load(path + filename))
    model.eval()
    return model

def generate_dataloader(dataset, batch_size):
    if(dataset == "MNIST"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('DeepRobust/image/data', train = False,
                        download = True,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading MNIST dataset.")

    elif(dataset == "CIFAR" or args.dataset == 'CIFAR10'):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('DeepRobust/image/data', train = False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Loading CIFAR10 dataset.")

    elif(dataset == "ImageNet"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('DeepRobust/image/data', train=False,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading ImageNet dataset.")
    return test_loader

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.", usage ='Use -h for more information.')

    parser.add_argument("--attack_method",
                        default = 'PGD',
                        help = "Choose a attack algorithm from: PGD(default), FGSM, LBFGS, CW, deepfool, onepixel, Nattack")
    parser.add_argument("--attack_model",
                        default = "CNN",
                        help = "Choose network structure from: CNN, ResNet")
    parser.add_argument("--path",
                        default = "./defense_model/",
                        help = "Type the path where the model is saved.")
    parser.add_argument("--file_name",
                        default = 'MNIST_CNN_epoch_20.pt',
                        help = "Type the file_name of the model that is to be attack. The model structure should be matched with the ATTACK_MODEL parameter.")
    parser.add_argument("--dataset",
                        default = 'MNIST',
                        help = "Choose a dataset from: MNIST(default), CIFAR(or CIFAR10), ImageNet")
    parser.add_argument("--epsilon", type = float, default = 0.3)
    parser.add_argument("--batch_num", type = int, default = 1000)
    parser.add_argument("--batch_size", type = int, default = 1000)
    parser.add_argument("--num_steps", type = int, default = 40)
    parser.add_argument("--step_size", type = float, default = 0.01)
    parser.add_argument("--random_targeted", type = bool, default = False,
                        help = "default: False. By setting this parameter be True, the program would random generate target labels for the input samples.")
    parser.add_argument("--device", default = 'cuda',
                        help = "Choose the device.")

    return parser.parse_args()


if __name__ == "__main__":
    # read arguments
    args = parameter_parser() # read argument and creat an argparse object

    # load model
    model = load_net(args.attack_model, args.file_name, args.path)

    print("===== START ATTACK =====")
    if(args.attack_method == "PGD"):
        from DeepRobust.image.attack.pgd import PGD
        test_loader = generate_dataloader(args.dataset, args.batch_size)
        attack_method = PGD(model, args.device)
        utils.tab_printer(args)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon)

    elif(args.attack_method == "FGSM"):
        from DeepRobust.image.attack.fgsm import FGM
        test_loader = generate_dataloader(args.dataset, args.batch_size)
        attack_method = FGM(model, args.device)
        utils.tab_printer(args)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon)

    elif(args.attack_method == "LBFGS"):
        from DeepRobust.image.attack.lbfgs import LBFGS
        try:
            if (args.batch_size >1):
                raise ValueError("batch_size shouldn't be larger than 1.")
        except ValueError:
            args.batch_size = 1

        utils.tab_printer(args)
        test_loader = generate_dataloader(args.dataset, args.batch_size)
        run_attack(attack_method, 1, args.batch_num, args.device, test_loader, random_targeted = args.random_targeted)

    elif(args.attack_method == "CW"):
        from DeepRobust.image.attack.cw import CarliniWagner
        attack_method = CarliniWagner(model, args.device)
        try:
            if (args.batch_size > 1):
                raise ValueError("batch_size shouldn't be larger than 1.")
        except ValueError:
            args.batch_size = 1

        utils.tab_printer(args)
        test_loader = generate_dataloader(args.dataset, args.batch_size)
        run_attack(attack_method, 1, args.batch_num, args.device, test_loader, random_targeted = args.random_targeted)

    elif(args.attack_method == "deepfool"):
        from DeepRobust.image.attack.deepfool import DeepFool
        attack_method = DeepFool(model, args.device)
        try:
            if (args.batch_size > 1):
                raise ValueError("batch_size shouldn't be larger than 1.")
        except ValueError:
            args.batch_size = 1

        utils.tab_printer(args)
        test_loader = generate_dataloader(args.dataset, args.batch_size)
        run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader)

    elif(args.attack_method == "onepixel"):
        from DeepRobust.image.attack.onepixle import OnePixel

    elif(args.attack_method == "Nattack"):
        pass

