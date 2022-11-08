"""
This is an implementation of [1]
References
---------
.. [1] Liu, F., Han, B., Liu, T., Gong, C., Niu, G., Zhou, M., & Sugiyama, M. (2021). 
Probabilistic margins for instance reweighting in adversarial training. 
Advances in Neural Information Processing Systems, 34, 23258-23269.
This implementation is based on their code: https://github.com/QizhouWang/MAIL
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
from deeprobust.image.utils import adjust_learning_rate
from deeprobust.image.attack.mail import MAIL
from deeprobust.image.attack.pgd import PGD

class MAIL(BaseDefense):
    """MAIL.
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

        # init model, Net() can be also used here for training
        optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum = self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma=0.1)

        for epoch in range(1, self.epochs + 1):
            print('Training epoch: ', epoch, flush=True)
            # MAIL training
            self.train(train_loader, optimizer, epoch)

            # evaluation on natural examples
            if epoch % self.test_freq == 0:
                self.test(test_loader)

            # save checkpoint
            if self.save_model:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'mail_model-nn-epoch{}.pt'.format(epoch)))
                    print('Model saved in ' + str(self.save_dir))

            scheduler.step()
    
    def parse_params(self,
                     epochs=120,
                     lr=0.1,
                     momentum=0.9,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     beta=6.0,
                     seed=1,
                     log_interval=100,
                     save_dir='./defense_models/mail',
                     save_freq=10,
                     bias=-1.5,
                     slope=1.0,
                     method='trades_mail',
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
        :param bias : float 
            - weighting bias term
        :param slope : float 
            - weighting bias term
        :param method : str 
            - parameter for basic method, choices = ['at_mail', 'trades_mail', 'mart_mail']
        """
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.beta = beta
        self.seed = seed 
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.bias = bias
        self.slope = slope
        self.method = method
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.distance_measure = distance_measure
        self.print_process = print_process
        self.test_epsilon = test_epsilon
        self.test_num_steps = test_num_steps
        self.test_step_size = test_step_size

    def test(self, model, device, test_loader):
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
            data_adv = self.adv_data(data, target, epoch)
            # calculate training loss
            loss = self.calculate_loss(data, data_adv, target)

            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def adv_data(self, data, target, epoch):
        """
        Generate input(adversarial) data for training.
        """
        adversary = MAIL(self.model)
        data_adv = adversary.generate(data, target, epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.step_size, method=self.method, 
                                      clip_max=self.clip_max, clip_min=self.clip_min, print_process=self.print_process, distance_measure=self.distance_measure)

        return data_adv

    def calculate_loss(self, x_natural, x_adv, y):
        """
        Calculate MAIL loss.
        """
        batch_size = len(x_natural)
        kl = nn.KLDivLoss(reduction='none')
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        if 'mail' not in self.method or epoch <= 75:
            if 'mart' in self.method:
                logits = self.model(x_natural)
                logits_adv = self.model(x_adv)

                adv_probs = F.softmax(logits_adv, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                nat_probs = F.softmax(logits, dim=1)
                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + float(self.beta) * loss_robust
            elif 'at' in self.method:
                logits_adv = self.model(x_adv)
                adv_probs = F.softmax(logits_adv, dim=1)
                loss = F.cross_entropy(logits_adv, y)
            elif 'trades' in self.method:
                logits = self.model(x_natural)
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(F.log_softmax(self.model(x_adv), dim=1), F.softmax(logits, dim=1))))
                loss = loss_natural + float(self.beta) * loss_robust
            else:
                raise RuntimeError('invalid method name')
        else:
            if 'mart' in self.method:
                logits = self.model(x_natural)
                logits_adv = self.model(x_adv)
                norm_weight = weight_assign(logits_adv, y, self.bias, self.slope)

                adv_probs = F.softmax(logits_adv, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(logits_adv, y, reduction='none') + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction='none')
                
                nat_probs = F.softmax(logits, dim=1)
                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                
                loss = (loss_adv * norm_weight).mean() + float(self.beta) * loss_robust
            elif 'at' in self.method:
                logits_adv = self.model(x_adv)
                norm_weight = weight_assign(logits_adv, y, self.bias, self.slope)

                adv_probs = F.softmax(logits_adv, dim=1)
                loss = F.cross_entropy(logits_adv, y, reduction='none')
                loss = (loss * norm_weight).mean()
            elif 'trades' in self.method:
                logits = self.model(x_natural)
                logits_adv = self.model(x_adv)
                norm_weight = weight_assign(logits_adv, y, self.bias, self.slope)

                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)), dim=1) * norm_weight)
                loss = loss_natural + float(self.beta) * loss_robust
            else:
                raise RuntimeError('invalid method name')

        return loss

def PM(logit, target):
    eye = torch.eye(10).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest=True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    
    return  probs_2nd - probs_GT

def weight_assign(logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    normalized_reweight = reweight * 3
    
    return normalized_reweight
