from abc import ABCMeta
import torch

class base_attack(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, device = 'cuda'):
        self.model = model
        self.device = device

    def generate(self, input, label, **kwargs):
        """
        :param x: input takes form (N, C, H, W)
        """
        return input

    def parse_params(self, **kwargs):
        return True

    def check_type_device(self, input, label):

        ################## devices
        if self.device == 'cuda':
            input = input.cuda()
            label = label.cuda()
            self.model = self.model.cuda()
        elif self.device == 'cpu':
            input = input.cpu()
            label = label.cpu()
            self.model = self.model.cpu()
        else:
            raise ValueError('Please input cpu or cuda')

        ################## data type
        if type(input).__name__ == 'Tensor':
            input = input.float()
            label = label.clone().detach().requires_grad_(True)
        elif type(x).__name__ == 'ndarray':
            input = input.astype('float')
            input = torch.tensor(input, requires_grad=True)
        else:
            raise ValueError('Input values only take numpy arrays or torch tensors')

        if type(label).__name__ == 'Tensor':
            label = label.long()
        elif type(label).__name__ == 'ndarray':
            label = label.astype('long')
            label = torch.tensor(y)
        else:
            raise ValueError('Input labels only take numpy arrays or torch tensors')


        #################### set init attributes
        self.input = input
        self.label = label

        return True


    def get_or_predict_lable(self, input):
        output = self.model(input)
        pred = output.argmax(dim=1, keepdim=True)
        return(pred)
