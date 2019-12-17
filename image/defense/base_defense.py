from abc import ABCMeta
import torch

class BaseDefense(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def parse_params(self, **kwargs):
        return True

    def generate(self, train_loader, test_loader, **kwargs):
        self.train_loader = train_loader
        self.test_loader = test_loader
        return 
    
    def train(self, train_loader, optimizer, epoch):
        return True

    def test(self, test_loader):
        return True
    def adv_data(self, model, data, target, **kwargs):
        return True

    def loss(self, output, target):
        return True

    def generate(self):
        return True
    
    def save_model(self):
        return True 
