"""
This is an implementation of [1]
References
---------
.. [1] Vivek, B. S., & Babu, R. V. (2020). 
Regularizers for single-step adversarial training. 
arXiv preprint arXiv:2002.00614.
This implementation is based on the code: https://github.com/val-iisc/SAT-Rx
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
from deeprobust.image.attack.sat import SAT
from deeprobust.image.attack.pgd import PGD

class SAT(BaseDefense):
    """
    SAT.
    """

    def __init__(self, model, device='cuda'):
        if not torch.cuda.is_available():
            print('CUDA not available, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)

    def generate(self, train_loader, test_loader, **kwargs):
        """Call this function to generate robust model.

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
            # TRADES training
            self.train(train_loader, optimizer, epoch)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'sat_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     method='R1',
                     Lambda=0.2,
                     train_ifgsm_steps=3,
                     epsilon_low=0.007,
                     epsilon_high=0.03,
                     rfgsm_alpha=0.007,
                     tau=0.6,
                     seed=1,
                     log_interval=100,
                     test_freq=10,
                     save_model=True,
                     save_dir='./defense_models/sat/',
                     save_freq=10,
                     clip_max=1.0,
                     clip_min=0.0,
                     print_process=False,
                     test_epsilon=0.031,
                     test_num_steps=20,
                     test_step_size=0.007):
        """Parameter parser.
        ----------
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
        :param method : str 
            - parameter for regularization method used in training, choices=['R1', 'R2', 'R3']
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.method = method
        self.Lambda = Lambda
        self.train_ifgsm_steps = train_ifgsm_steps
        self.epsilon_low= epsilon_low
        self.epsilon_high = epsilon_high
        self.rfgsm_alpha = rfgsm_alpha
        self.tau = tau
        self.seed = seed 
        self.log_interval = log_interval
        self.test_freq = test_freq
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_freq = save_freq
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

    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)

            # generate adversarial examples
            if self.method == "R1":
                data_adv = self.adv_data_1(data, target, len(data))
                # calculate training loss
                loss = self.calculate_loss_1(data_adv, target)
            elif self.method == "R2":
                data_adv = self.adv_data_2(data, target)
                # calculate training loss
                loss = self.calculate_loss_2(data, data_adv, target)
            elif self.method == "R3":
                data_adv = self.adv_data_3(data, target)
                # calculate training loss
                loss = self.calculate_loss_3(data, data_adv, target)
            else:
                raise RuntimeError('invalid regularization method')

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data_1(self, data, target, batch_size):
        """
        Generate input(adversarial) data for training.
        """
        adversary = SAT(self.model)
        #FGSM samples for m samples in the mini-batch
        data_adv_1 = adversary.generate(data, target, epsilon=self.epsilon, num_steps=1, method='R1', clip_max=self.clip_max, clip_min=self.clip_min, 
                                        print_process=self.print_process, distance_measure=self.distance_measure)

        #I-FGSM adversarial sample corresponding to last sample in the mini-batch
        data_adv_2  = adversary.generate(data[batch_size-1:,:,:,:], target[batch_size-1:], epsilon=self.epsilon, num_steps=self.train_ifgsm_steps, 
                                         method='R1', clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=self.distance_measure)

        data_adv = torch.cat((data_adv_1, data_adv_2), 0)

        return data_adv

    def calculate_loss_1(self, x_adv, y):
        """
        Calculate SAT loss.
        """
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate nature loss
        adv_logits = self.model(x_adv)
        B = len(adv_logits)

        CE_loss = F.cross_entropy(adv_logits[:B-1,:], y)

        REG_loss = l2_square(adv_logits[B-2:B-1,:], adv_logits[B-1:,:])

        loss = CE_loss + self.Lambda * REG_loss
        
        return loss

    def adv_data_2(self, data, target):
        """
        Generate input(adversarial) data for training.
        """
        adversary = SAT(self.model)
        data_adv = adversary.generate(data, target, epsilon=self.epsilon, method='R2', eps_iter=self.rfgsm_alpha, clip_max=self.clip_max,
                                      clip_min=self.clip_min, print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss_2(self, x_adv, y):
        """
        Calculate SAT loss.
        """
        batch_size = len(x_adv)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate nature loss
        adv_logits = self.model(x_adv)
        B = len(adv_logits)

        # Cross-entropy loss on FGSM samples only
        CE_loss = F.cross_entropy(adv_logits[:int(B/2),:], y)

        REG_loss = l2_square(adv_logits[:int(B/2),:], adv_logits[int(B/2):,:])

        loss = CE_loss + self.Lambda * REG_loss
        
        return loss

    def adv_data_3(self, data, target):
        """
        Generate input(adversarial) data for training.
        """
        adversary = SAT(self.model)
        data_adv = adversary.generate(data, target, method='R3', epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high, alpha=self.rfgsm_alpha, 
                                      clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss_3(self, x_adv, y):
        """
        Calculate SAT loss.
        """
        batch_size = len(x_adv)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate nature loss
        adv_logits = self.model(x_adv)
        B = len(adv_logits)

        # Cross-entropy loss on FGSM samples only
        loss_eps_low = F.cross_entropy(adv_logits[0:int(B/2),:], y)
        loss_eps_high = F.cross_entropy(adv_logits[int(B/2):,:], y)

        loss = loss_eps_high + self.Lambda * F.relu(loss_eps_low - self.tau * loss_eps_high)
        
        return loss

# Return Euclidean distance (**2) between 2 vectors
def l2_square(x, y):
    diff = x - y
    diff = diff * diff
    diff = diff.sum(1)
    diff = diff.mean(0)

    return diff
