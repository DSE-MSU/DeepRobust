"""
https://github.com/ferjad/Universal_Adversarial_Perturbation_pytorch
Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>

"""
from deeprobust.image.attack import deepfool
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd.gradcheck import zero_gradients
import math
from PIL import Image
import torchvision.models as models
import sys
import random
import time
from tqdm import tqdm

def get_model(model,device):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model =='resnet18':
        net = models.resnet18(pretrained=True)

    net.eval()
    net=net.to(device)
    return net

def data_input_init(xi):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])

    return (mean,std,transform)

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    if p==np.inf:
        v=torch.clamp(v,-xi,xi)
    else:
        v=v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v

def get_fooling_rate(data_list,v,model, device):
    f = data_input_init(0)[2]
    num_images = len(data_list)

    fooled=0.0

    for name in tqdm(data_list):
        image = Image.open(name)
        image = tf(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        _, pred = torch.max(model(image),1)
        _, adv_pred = torch.max(model(image+v),1)
        if(pred!=adv_pred):
            fooled+=1

    # Compute the fooling rate
    fooling_rate = fooled/num_images
    print('Fooling Rate = ', fooling_rate)
    for param in model.parameters():
        param.requires_grad = False

    return fooling_rate,model

def universal_adversarial_perturbation(dataloader, model, device, xi=10, delta=0.2, max_iter_uni = 10, p=np.inf,
                                       num_classes=10, overshoot=0.02, max_iter_df=10,t_p = 0.2):
    """universal_adversarial_perturbation.

    Parameters
    ----------
    dataloader :
        dataloader
    model :
        target model
    device :
        device
    xi :
        controls the l_p magnitude of the perturbation
    delta :
        controls the desired fooling rate (default = 80% fooling rate)
    max_iter_uni :
        maximum number of iteration (default = 10*num_images)
    p :
        norm to be used (default = np.inf)
    num_classes :
        num_classes (default = 10)
    overshoot :
        to prevent vanishing updates (default = 0.02)
    max_iter_df :
        maximum number of iterations for deepfool (default = 10)
    t_p :
        truth percentage, for how many flipped labels in a batch. (default = 0.2)

    Returns
    -------
        the universal perturbation matrix.
    """
    time_start = time.time()
    mean, std,tf = data_input_init(xi)
    v = torch.zeros(1,3,224,224).to(device)
    v.requires_grad_()

    fooling_rate = 0.0
    num_images =  len(data_list)
    itr = 0

    while fooling_rate < 1-delta and itr < max_iter_uni:

        # Iterate over the dataset and compute the purturbation incrementally

        for i,(img, label) in enumerate(dataloader):
            _, pred = torch.max(model(img),1)
            _, adv_pred = torch.max(model(img+v),1)

            if(pred == adv_pred):
                perturb = deepfool(model, device)
                _ = perturb.generate(img+v, num_classed = num_classed, overshoot = overshoot, max_iter = max_iter_df)
                dr, iter = perturb.getpurb()
                if(iter<max_iter_df-1):
                    v = v + torch.from_numpy(dr).to(device)
                    v = proj_lp(v,xi,p)

            if(k%10==0):
                print('Norm of v: '+str(torch.norm(v).detach().cpu().numpy()))

        fooling_rate,model = get_fooling_rate(data_list,v,model, device)
        itr = itr + 1

    return v
