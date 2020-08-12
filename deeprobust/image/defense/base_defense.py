from abc import ABCMeta
import torch

class BaseDefense(object):
    """
    Defense base class.
    """


    __metaclass__ = ABCMeta

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def parse_params(self, **kwargs):
        """
        Parse user defined parameters
        """
        return True

    def generate(self, train_loader, test_loader, **kwargs):
        """generate.

        Parameters
        ----------
        train_loader :
            training data
        test_loader :
            testing data
        kwargs :
            user defined parameters
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        return 
    
    def train(self, train_loader, optimizer, epoch):
        """train.

        Parameters
        ----------
        train_loader :
            training data
        optimizer :
            training optimizer
        epoch :
            training epoch
        """
        return True

    def test(self, test_loader):
        """test.

        Parameters
        ----------
        test_loader :
            testing data
        """
        return True
    def adv_data(self, model, data, target, **kwargs):
        """
        Generate adversarial examples for adversarial training.
        Overide this function to generate customize adv examples.

        Parameters
        ----------
        model :
            victim model
        data :
            original data
        target :
            target labels
        kwargs :
            parameters
        """
        return True

    def loss(self, output, target):
        """
        Calculate training loss. 
        Overide this function to customize loss.
        
        Parameters
        ----------
        output :
            model outputs
        target :
            true labels
        """
        return True

    def generate(self):
        return True
    
    def save_model(self):
        """
        Save model.
        """
        return True 
