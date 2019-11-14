import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
import os

from DeepRobust.image.attack.deepfool import DeepFool


#load pretrained model
net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

#load original example
im_orig = Image.open('test_im2.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])(im_orig)

print(im)
Attack1 = DeepFool(net, device = 'cuda')
r, loop_i, label_orig, label_pert, pert_image = Attack1.generate(im)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

# A = torch.clamp(A, 0, 255)

transf = transforms.Compose([transforms.Lambda(clip)])


im_orig = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])(im_orig)

#print(pert_image)
im_pert = transf(pert_image.cpu()[0])
im_diff = im_pert - im_orig

im_orig = transforms.ToPILImage()(im_orig).convert('RGB')
im_pert = transforms.ToPILImage()(im_pert).convert('RGB')
im_diff = transforms.ToPILImage()(im_diff).convert('RGB')

plt.imshow(im_pert)
plt.savefig('deepfool_pert.png')
plt.imshow(im_orig)
plt.savefig('deepfool_orig.png')
plt.imshow(im_diff)
plt.savefig('deepfool_diff.png')