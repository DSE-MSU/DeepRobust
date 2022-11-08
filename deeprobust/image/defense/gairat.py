"""
This is an implementation of [1]
References
---------
.. [1] Zhang, J., Zhu, J., Niu, G., Han, B., Sugiyama, M., & Kankanhalli, M. (2020). 
Geometry-aware instance-reweighted adversarial training. 
arXiv preprint arXiv:2010.01736.
This implementation is based on their code: https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from deeprobust.image.defense.base_defense import BaseDefense
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.attack.gairat import GAIRAT
from deeprobust.image.attack.pgd import PGD


class GAIRAT(BaseDefense):
    """GAIRAT.
    """

    def __init__(self, model, device = 'cuda'):
        if not torch.cuda.is_available():
            print('CUDA not available, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)

    def generate(self, train_loader, test_loader, **kwargs):
        """generate robust model.
        Parameters
        ----------
        train_loader :
            train_loader
        test_loader :
            test_loader
        kwargs :
            kwargs
        """
        
        self.parse_params(**kwargs)
        
        torch.manual_seed(self.seed)

        # initialize model, Net() can be also used here for training
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.lr_schedule == 'superconverge':
            lr_schedule_func = lambda t: np.interp([t], [0, self.epochs * 2 // 5, self.epochs], [0, self.lr_max, 0])[0]
        elif self.lr_schedule == 'piecewise':
            def lr_schedule_func(t):
                if self.epochs >= 110:
                    # Train Wide-ResNet
                    if t / self.epochs < 0.5:
                        return self.lr_max
                    elif t / self.epochs < 0.75:
                        return self.lr_max / 10.
                    elif t / self.epochs < (11/12):
                        return self.lr_max / 100.
                    else:
                        return self.lr_max / 200.
                else:
                    # Train ResNet
                    if t / self.epochs < 0.3:
                        return self.lr_max
                    elif t / self.epochs < 0.6:
                        return self.lr_max / 10.
                    else:
                        return self.lr_max / 100.
        elif self.lr_schedule == 'linear':
            lr_schedule_func = lambda t: np.interp([t], [0, self.epochs // 3, self.epochs * 2 // 3, self.epochs], [self.lr_max, self.lr_max, self.lr_max / 10, self.lr_max / 100])[0]
        elif self.lr_schedule == 'onedrop':
            def lr_schedule_func(t):
                if t < self.lr_drop_epoch:
                    return self.lr_max
                else:
                    return self.lr_one_drop
        elif self.lr_schedule == 'multipledecay':
            def lr_schedule_func(t):
                return self.lr_max - (t//(self.epochs//10))*(self.lr_max/10)
        elif self.lr_schedule == 'cosine': 
            def lr_schedule_func(t): 
                return self.lr_max * 0.5 * (1 + np.cos(t / self.epochs * np.pi))
        else:
            raise RuntimeError('invalid lr schedule function')


        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            Lambda = adjust_Lambda(epoch, self.epochs, self.Lambda, self.Lambda_max, self.Lambda_schedule)

            # adversarial training
            self.train(train_loader, optimizer, epoch, Lambda, lr_schedule_func)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'gairat_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     seed=1,
                     log_interval=100,
                     save_dir='./defense_models/gairat/',
                     save_freq=10,
                     weight_assignment_function='Tanh',
                     begin_epoch=60,
                     Lambda='-1.0',
                     Lambda_max=float('inf'),
                     Lambda_schedule='fixed',
                     lr_schedule='piecewise',
                     lr_max=0.1,
                     lr_one_drop=0.01,
                     lr_drop_epoch=100,
                     category='trades',
                     loss_fn='cent',
                     rand_init=True,
                     clip_max=1.0,
                     clip_min=0.0,
                     distance_measure='l_inf',
                     print_process=False,
                     test_epsilon=0.031,
                     test_num_steps=20,
                     test_step_size=0.007):
        """
        :param epoch : int 
            - pgd training epoch
        :param save_dir : str 
            - directory path to save model
        :param epsilon : float 
            - perturb constraint of pgd adversary example used to train defense model
        :param num_steps : int 
            - the perturb 
        :param perturb_step_size : float 
            - step_size 
        :param lr : float 
            - learning rate for adversary training process
        :param momentum : float 
            - parameter for optimizer in training process
        :param weight_assignment_function : str 
            - choices = ['Discrete','Sigmoid','Tanh']
        :param begin_epoch : int 
            - when to use GAIR
        :param Lambda : str 
            - parameter for GAIR
        :param Lambda_max : float 
            - max Lambda
        :param Lambda_schedule : str
            - choices = ['linear', 'piecewise', 'fixed']
        :param lr-schedule : str
            - choices = ['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine']
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed 
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.weight_assignment_function = weight_assignment_function
        self.begin_epoch = begin_epoch
        self.Lambda = Lambda
        self.Lambda_max = Lambda_max
        self.Lambda_schedule = Lambda_schedule
        self.lr_schedule = lr_schedule
        self.lr_max = lr_max
        self.lr_one_drop = lr_one_drop
        self.lr_drop_epoch = lr_drop_epoch
        self.category = category
        self.loss_fn = loss_fn
        self.rand_init = rand_init
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.distance_measure = distance_measure
        self.print_process = print_process
        self.test_epsilon = test_epsilon
        self.test_num_steps = test_num_steps
        self.test_step_size = test_step_size

    def test(self, test_loader):
        self.model.eval()

        clean_test_loss = 0
        adv_test_loss = 0
        clean_correct = 0
        adv_correct = 0

        test_adversary = PGD(self.model)
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            clean_test_loss += F.cross_entropy(clean_output, target, reduction='sum').item()
            clean_pred = clean_output.max(1, keepdim=True)[1]
            clean_correct += clean_pred.eq(target.view_as(clean_pred)).sum().item()

            # robust evaluation using PGD-20
            data_adv = test_adversary.generate(data, target, epsilon=self.test_epsilon, num_steps=self.test_num_steps, step_size=self.test_step_size, 
                                               clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=args.distance_measure)
            adv_output = self.model(data_adv)
            adv_test_loss += F.cross_entropy(adv_output, target, reduction='sum').item()
            adv_pred = adv_output.max(1, keepdim=True)[1]
            adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()

        lean_test_loss /= len(test_loader.dataset)
        adv_test_loss /= len(test_loader.dataset)

        print('Test: Clean loss: {:.4f}, Clean accuracy: {}/{} ({:.2f}%)'.format(
        clean_test_loss, clean_correct, len(test_loader.dataset),
        100. * clean_correct / len(test_loader.dataset)), flush=True)
        print('Test: Robust loss: {:.4f}, Robust accuracy: {}/{} ({:.2f}%)'.format(
        adv_test_loss, adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)), flush=True)

    def train(self, train_loader, optimizer, epoch, Lambda, lr_schedule_func):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)

            # calculate robust loss
            data_adv, Kappa = self.adv_data(data, target)
            # calculate training loss
            loss = self.calculate_loss(data, data_adv, target, Lambda, Kappa, lr_schedule_func, epoch)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data(self, data, target):
        """
        Generate input(adversarial) data for training.
        """

        adversary = GAIRAT(self.model)
        data_adv, Kappa = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size, category=self.category, 
                                             loss_fn=self.loss_fn, rand_init=self.rand_init, clip_max=self.clip_max, clip_min=self.clip_min, 
                                             print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv, Kappa

    def calculate_loss(self, x_natural, x_adv, y, Lambda, Kappa, lr_schedule, epoch):
        """
        Calculate Trades loss.
        """
        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        self.model.train()

        lr = lr_schedule(epoch)
        optimizer.param_groups[0].update(lr=lr)
        # zero gradient
        optimizer.zero_grad()

        logits_adv = self.model(x_adv)

        if (epoch) >= self.begin_epoch:
            Kappa = Kappa.to(self.device)
            loss = nn.CrossEntropyLoss(reduce=False)(logits_adv, y)
            # calculate weight assignment according to geometry value
            normalized_reweight = GAIR(self.num_steps, Kappa, Lambda, self.weight_assignment_function)
            loss = loss.mul(normalized_reweight).mean()
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logits_adv, y)

        return loss

# adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch, total_epochs, Lambda, Lambda_max, Lambda_schedule):
    Lam = float(Lambda)
    if total_epochs >= 110:
        # train Wide-ResNet
        Lambda = Lambda_max
        if Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = Lambda_max - (epoch/total_epochs) * (Lambda_max - Lam)
        elif Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # train ResNet
        Lambda = Lambda_max
        if Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = Lambda_max - (epoch/total_epochs) * (Lambda_max - Lam)
        elif Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam

    return Lambda

def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight

