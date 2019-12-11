#__init__.py
from DeepRobust.image.netmodels import CNN
from DeepRobust.image.netmodels import resnet
from DeepRobust.image.netmodels import YOPOCNN
from DeepRobust.image.netmodels import train_model

__all__ = ['CNNmodel','resnet','YOPOCNN','train_model']
