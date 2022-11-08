"""
This is an implementation of [1]
References
---------
.. [1] Balaji, Y., Goldstein, T., & Hoffman, J. (2019). 
Instance adaptive adversarial training: Improved accuracy tradeoffs in neural nets. 
arXiv preprint arXiv:1910.08051.
This implementation is based on their code: https://github.com/yogeshbalaji/Instance_Adaptive_Adversarial_Training
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
from deeprobust.image.attack.iaat import IAAT
from deeprobust.image.attack.pgd import PGD

class IAAT(BaseDefense):
    """
    IAAT.
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
        optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum = self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma=0.1)

        self.epsilon_memory = torch.FloatTensor(len(train_dataset)).zero_().to(self.device)

        for epoch in range(1, self.epochs + 1):
            print('Training epoch: ', epoch, flush=True)
            # IAAT training
            self.train(train_loader, optimizer, epoch)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'trade_model-nn-epoch{}.pt'.format(epoch)))
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
                     test_freq=10,
                     save_dir='./defense_models/iaat/',
                     save_freq=10,
                     warmup_epoch=10,
                     gamma=1.9/255,
                     beta=0.1,
                     attack_lr=0.007,
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
        :param warmup_epoch : int 
            - parameter for warm up training (10 epochs for CIFAR-10/100)
        :param gamma : float 
            - parameter for discretization
        :param beta : float 
            - parameter for exponential averaging factor
        :param attack_lr : float 
            - parameter for attack learning rate
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
        self.warmup_epoch = warmup_epoch
        self.gamma = gamma
        self.beta = beta
        self.attack_lr = attack_lr
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
        for batch_idx, (data, target, indices) in enumerate(train_loader):
            
            optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)

            # generate adversarial examples
            data_adv = self.adv_data(data, target, indices, epoch)
            # calculate training loss
            loss = self.calculate_loss(data_adv, target)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    def adv_data(self, data, target, indices, epoch):
        """
        Generate input(adversarial) data for training.
        """

        epsilon_arr = self.epsilon_select(data, target, indices, epoch)
        attack_lr_arr = torch.clamp(epsilon_arr / (0.5 * self.num_steps), min=self.attack_lr)

        adversary = IAAT(self.model)
        data_adv = adversary.generate(data, target, num_steps=self.num_steps, attack_lr_arr=attack_lr_arr, attack_eps_arr=epsilon_arr, 
                                      clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss(self, x_adv, y):
        """
        Calculate IAAT loss.
        """
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        logits_adv = model(x_adv)
        loss = F.cross_entropy(logits_adv, y)
        
        return loss

    def epsilon_select(self, data, target, indices, cur_epoch):
        # self.model.eval()
        with torch.no_grad():
            logits = self.model(data)
            _, pred = torch.max(logits, dim=1)
            correct_preds_clean = (pred == target).float()

        if cur_epoch < self.warmup_epoch:
            epsilon = torch.zeros(data.size(0)).fill_(self.epsilon).to(self.device)
            epsilon = epsilon * correct_preds_clean
        else:
            epsilon_prev = self.epsilon_memory[indices]
            epsilon_low = epsilon_prev - self.gamma
            epsilon_cur = epsilon_prev
            epsilon_high = epsilon_prev + self.gamma
            attack_lr_cur = torch.clamp(epsilon_cur / (0.5 * self.num_steps), min=self.attack_lr)
            attack_lr_high = torch.clamp(epsilon_high / (0.5 * self.num_steps), min=self.attack_lr)

            adv_adjust = IAAT(self.model)
            input_cur = adv_adjust.generate(data, target, num_steps=self.num_steps, attack_lr_arr=attack_lr_cur, attack_eps_arr=epsilon_cur)
            input_high = adv_adjust.generate(data, target, num_steps=self.num_steps, attack_lr_arr=attack_lr_high, attack_eps_arr=epsilon_high)

            with torch.no_grad():
                logits_cur = self.model(input_cur)
                logits_high = self.model(input_high)
                _, logits_cur = torch.max(logits_cur, dim=1)
                _, logits_high = torch.max(logits_high, dim=1)

                pred_cur = (logits_cur == target).float()
                pred_high = (logits_high == target).float()

                epsilon = pred_high * epsilon_high + (1 - pred_high) * pred_cur * epsilon_cur + (1 - pred_high) * (1 - pred_cur) * epsilon_low
                epsilon = epsilon * correct_preds_clean
                epsilon = torch.clamp(epsilon, min=self.clip_min)
                epsilon = epsilon * self.beta + epsilon_prev * (1 - self.beta)
        # Updating memory
        self.epsilon_memory[indices] = epsilon

        return epsilon

