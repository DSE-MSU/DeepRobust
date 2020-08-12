from abc import ABCMeta
import torch

class BaseAttack(object):
    """
    Attack base class.
    """

    __metaclass__ = ABCMeta

    def __init__(self, model, device = 'cuda'):
        self.model = model
        self.device = device

    def generate(self, image, label, **kwargs):
        """
        Overide this function for the main body of attack algorithm.

        Parameters
        ----------
        image :
            original image
        label :
            original label
        kwargs :
            user defined parameters
        """
        return input

    def parse_params(self, **kwargs):
        """
        Parse user defined parameters.
        """
        return True

    def check_type_device(self, image, label):
        """
        Check device, match variable type to device type.

        Parameters
        ----------
        image :
            image
        label :
            label
        """

        ################## devices
        if self.device == 'cuda':
            image = image.cuda()
            label = label.cuda()
            self.model = self.model.cuda()
        elif self.device == 'cpu':
            image = image.cpu()
            label = label.cpu()
            self.model = self.model.cpu()
        else:
            raise ValueError('Please input cpu or cuda')

        ################## data type
        if type(image).__name__ == 'Tensor':
            image = image.float()
            image = image.float().clone().detach().requires_grad_(True)
        elif type(x).__name__ == 'ndarray':
            image = image.astype('float')
            image = torch.tensor(image, requires_grad=True)
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
        self.image = image
        self.label = label

        return True

    def get_or_predict_lable(self, image):
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        return(pred)
