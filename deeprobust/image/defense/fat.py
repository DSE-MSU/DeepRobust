"""
This is an implementation of [1]
References
---------
.. [1] Zhang, J., Xu, X., Han, B., Niu, G., Cui, L., Sugiyama, M., & Kankanhalli, M. (2020, November). 
Attacks which do not kill training make adversarial learning stronger. 
In International conference on machine learning (pp. 11278-11287). PMLR.
This implementation is based on their code: https://github.com/zjfheart/Friendly-Adversarial-Training
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
from deeprobust.image.attack.pgd import FAT
from deeprobust.image.attack.pgd import PGD

class FAT(BaseDefense):
    """FAT.
    """

    def __init__(self, model, device='cuda'):
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma=0.1)

        for epoch in range(1, self.epochs + 1):
            print('Training epoch: ', epoch, flush=True)
            # FAT training
            self.train(train_loader, optimizer, epoch, adjust_tau(epoch, self.dynamictau))

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'fat_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     seed=1,
                     log_interval=100,
                     test_freq=1,
                     save_model=True,
                     save_dir='./defense_models/fat/',
                     save_freq=10,
                     dynamictau=True,
                     tau=0,
                     omega=0.001,
                     loss_func='pgd',
                     rand_init=True,
                     randominit_type='uniform_randominit',
                     beta = 6.0,
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
        :param dynamictau : bool 
            - parameter for choosing whether to use dynamic tau
        :param tau : int 
            - step tau
        :param loss_func : str
        	- parameter for choosing loss function used in training, choices=['pgd', 'trades', 'mart']
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed 
        self.log_interval = log_interval
        self.test_freq = test_freq
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.dynamictau = dynamictau
        self.tau = tau
        self.omega = omega
        self.loss_func = loss_func
        self.rand_init = rand_init
        self.randominit_type=randominit_type
        self.beta = beta
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

    def train(self, train_loader, optimizer, epoch, tau):
        self.model.train()
        
        # loss function used for generating adversarial examples
        if self.loss_func == 'trades':
            self.loss_fn = 'kl'
        else:
            self.loss_fn = 'cent' 

        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)

            # generate adversarial examples
            data_adv, target_adv, data_natural = self.adv_data(data, target, tau)
            # calculate training loss
            loss = self.calculate_loss(data_natural, data_adv, target_adv)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data(self, data, target, tau):
        """
        Generate input(adversarial) data for training.
        """

        adversary = FAT(self.model)
        data_adv, target_adv, data_natural = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size, 
                                                                loss_fn=self.loss_fn, tau=tau, omega=self.omega, rand_init=self.rand_init, 
                                                                randominit_type=self.randominit_type, clip_max=self.clip_max, clip_min=self.clip_min, 
                                                                print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv, target_adv, data_natural

    def calculate_loss(self, x_natural, x_adv, y):
        """
        Calculate FAT loss.
        """
        batch_size = len(x_natural)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        if self.loss_func == 'trades':
            criterion_kl = nn.KLDivLoss(reduction='sum')
    
            # calculate nature loss
            logits = self.model(x_natural)
            loss_natural = F.cross_entropy(logits, y)
            # calculate robust loss
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                                            F.softmax(self.model(x_natural), dim=1))
            loss = loss_natural + self.beta * loss_robust
        elif self.loss_func == 'mart':
            criterion_kl = nn.KLDivLoss(reduction='none')

            logits = self.model(x_natural)
            logits_adv = self.model(x_adv)

            adv_probs = F.softmax(logits_adv, dim=1)
            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) 
                                                         * (1.0000001 - true_probs))
            loss = loss_adv + self.beta * loss_robust
        elif self.loss_func == 'pgd':
            logits_adv = self.model(x_adv)
            loss = F.cross_entropy(logits_adv, y)
        else:
            raise RuntimeError('invalid loss function')
        
        return loss

def adjust_tau(epoch, dynamictau):
    tau = self.tau
    if dynamictau:
        if epoch < 50:
            tau = 0
        elif epoch < 90:
            tau = 1
        else:
            tau = 2

    return tau
