from abc import ABCMeta
import torch

class BaseDefense(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def parse_params(self, **kwargs):
        return True

    def generate(self, **kwargs):
        return True
    
    def train(self, train_loader, optimizer, epoch):
        model.train()
        correct = 0
        bs = train_loader.batch_size

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            train_data = self.adv_data(model, data, target)
            AdvExArray = adversary.generate(data, target, epsilon = 0.3, num_steps = 40)

            output = model(AdvExArray)
            loss = self.loss(output, target)
            loss.backward()
            
            optimizer.step()

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            #print every 10
            if batch_idx % 10 == 0:
               self.print_accuracy()
              
            correct = 0

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
