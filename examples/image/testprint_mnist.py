import matplotlib.pyplot as plt 
import torch
from torchvision import datasets
import numpy as py

dataset=datasets.MNIST('../data')
x=dataset.data[8888]
print(x.shape)
#print("pic info",x)
#x.save("imageexample.png")