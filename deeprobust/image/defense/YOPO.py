"""
This is an implementation of adversarial training variant: YOPO.
References
----------
.. [1] Zhang, D., Zhang, T., Lu, Y., Zhu, Z., & Dong, B. (2019).
You only propagate once: Painless adversarial training using maximal principle.
arXiv preprint arXiv:1905.00877.

.. [2] Original code: https://github.com/a1600012888/YOPO-You-Only-Propagate-Once
"""

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import optim

from collections import OrderedDict
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict
import numpy as np
import argparse
import json
import math

import os

from deeprobust.image.netmodels import YOPOCNN
from deeprobust.image import utils
from deeprobust.image.attack import YOPOpgd
from deeprobust.image.defense.base_defense import BaseDefense

import time
from tensorboardX import SummaryWriter

class PieceWiseConstantLrSchedulerMaker(object):

    def __init__(self, milestones:List[int], gamma:float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)


class IPGDAttackMethodMaker(object):

    def __init__(self, eps, sigma, nb_iters, norm, mean, std):
        self.eps = eps
        self.sigma = sigma
        self.nb_iters = nb_iters
        self.norm = norm
        self.mean = mean
        self.std = std

    def __call__(self, DEVICE):
        return YOPOpgd.FASTPGD(self.eps, self.sigma, self.nb_iters, self.norm, DEVICE, self.mean, self.std)

def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans

class AvgMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num

def load_checkpoint(file_name, net = None, optimizer = None, lr_scheduler = None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        check_point = torch.load(file_name)
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['state_dict'])
        if optimizer is not None:
            print('Loading optimizer state dict')
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        if lr_scheduler is not None:
            print('Loading lr_scheduler state dict')
            lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        return check_point['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


def make_symlink(source, link_name):
    if os.path.exists(link_name):
        #print("Link name already exist! Removing '{}' and overwriting".format(link_name))
        os.remove(link_name)
    if os.path.exists(source):
        os.symlink(source, link_name)
        return
    else:
        print('Source path not exists')
    #print('SymLink Wrong!')

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof = 1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        H = torch.sum(y * p)
        return H

class CrossEntropyWithWeightPenlty(_Loss):
    def __init__(self, module, DEVICE, reg_cof = 1e-4):
        super(CrossEntropyWithWeightPenlty, self).__init__()

        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.module = module

    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = cal_l2_norm(self.module)

        loss = cross_loss + self.reg_cof * weight_loss
        return loss

def cal_l2_norm(layer: torch.nn.Module):
 loss = 0.
 for name, param in layer.named_parameters():
     if name == 'weight':
         loss = loss + 0.5 * torch.norm(param,) ** 2

 return loss

class FastGradientLayerOneTrainer(object):

    def __init__(self, Hamiltonian_func, param_optimizer,
                    inner_steps=2, sigma = 0.008, eps = 0.03):
        self.inner_steps = inner_steps
        self.sigma = sigma
        self.eps = eps
        self.Hamiltonian_func = Hamiltonian_func
        self.param_optimizer = param_optimizer

    def step(self, inp, p, eta):
        p = p.detach()

        for i in range(self.inner_steps):
            tmp_inp = inp + eta
            tmp_inp = torch.clamp(tmp_inp, 0, 1)
            H = self.Hamiltonian_func(tmp_inp, p)

            eta_grad_sign = torch.autograd.grad(H, eta, only_inputs=True, retain_graph=False)[0].sign()

            eta = eta - eta_grad_sign * self.sigma

            eta = torch.clamp(eta, -1.0 * self.eps, self.eps)
            eta = torch.clamp(inp + eta, 0.0, 1.0) - inp
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        #self.param_optimizer.zero_grad()

        yofo_inp = eta + inp
        yofo_inp = torch.clamp(yofo_inp, 0, 1)

        loss = -1.0 * self.Hamiltonian_func(yofo_inp, p)

        loss.backward()
        #self.param_optimizer.step()
        #self.param_optimizer.zero_grad()

        return yofo_inp, eta

def eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)

            with torch.no_grad():
                pred = net(adv_inp)
                acc = torch_accuracy(pred, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if AttackMethod is not None else 0
    return clean_accuracy.mean, adv_acc


class SGDOptimizerMaker(object):

    def __init__(self, lr = 0.1, momentum = 0.9, weight_decay = 1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, params):
        return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

def main():
    num_epochs = 40
    val_interval = 1
    weight_decay = 5e-4

    inner_iters = 10
    K = 5
    sigma = 0.01
    eps = 0.3
    lr = 1e-2
    momentum = 0.9
    create_optimizer = SGDOptimizerMaker(lr =1e-2 / K, momentum = 0.9, weight_decay = weight_decay)

    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [30, 35, 39], gamma = 0.1)

    create_loss_function = None

    create_attack_method = None

    create_evaluation_attack_method = IPGDAttackMethodMaker(eps = 0.3, sigma = 0.01, nb_iters = 40, norm = np.inf,
                                      mean=torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                                      std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir',default = "./trained_models")
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size')
    parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
    parser.add_argument('-adv_coef', default=1.0, type = float,
                        help = 'Specify the weight for adversarial loss')
    parser.add_argument('--auto-continue', default=False, action = 'store_true',
                        help = 'Continue from the latest checkpoint')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = YOPOCNN.Net()
    net.to(DEVICE)
    criterion = CrossEntropyWithWeightPenlty(net.other_layers, DEVICE, weight_decay)#.to(DEVICE)
    optimizer = create_optimizer(net.other_layers.parameters())
    lr_scheduler = create_lr_scheduler(optimizer)

    Hamiltonian_func = Hamiltonian(net.layer_one, weight_decay)
    layer_one_optimizer = optim.SGD(net.layer_one.parameters(), lr = lr_scheduler.get_lr()[0], momentum=0.9, weight_decay=5e-4)
    lyaer_one_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(layer_one_optimizer,
                                                                    milestones = [15, 19], gamma = 0.1)
    LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, layer_one_optimizer,
                                                inner_iters, sigma, eps)

    ds_train = utils.create_train_dataset(args.batch_size)
    ds_val = utils.create_test_dataset(args.batch_size)

    EvalAttack = create_evaluation_attack_method(DEVICE)

    now_epoch = 0

    if args.auto_continue:
        args.resume = os.path.join(args.model_dir, 'last.checkpoint')
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer,lr_scheduler)

    now_train_time = 0
    while True:
        if now_epoch > num_epochs:
            break
        now_epoch = now_epoch + 1

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(now_epoch, num_epochs,
                                                                        lr_scheduler.get_lr()[0])
        s_time = time.time()

        #train
        acc, yopoacc = train_one_epoch(net, ds_train, optimizer, eps, criterion, LayerOneTrainer, K,
                        DEVICE, descrip_str)

        now_train_time = now_train_time + time.time() - s_time
        tb_train_dic = {'Acc':acc, 'YoPoAcc':yopoacc}
        print(tb_train_dic)

        lr_scheduler.step()
        lyaer_one_optimizer_lr_scheduler.step()
        utils.save_checkpoint(now_epoch, net, optimizer, lr_scheduler,
                        file_name = os.path.join(args.model_dir, 'epoch-{}.checkpoint'.format(now_epoch)))

def train_one_epoch(net, batch_generator, optimizer, eps,
                    criterion, LayerOneTrainner, K,
                    DEVICE=torch.device('cuda:0'),descrip_str='Training'):
    '''
    :param attack_freq:  Frequencies of training with adversarial examples. -1 indicates natural training
    :param AttackMethod: the attack method, None represents natural training
    :return:  None    #(clean_acc, adv_acc)
    '''
    net.train()
    pbar = tqdm(batch_generator)
    yofoacc = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        eta = torch.FloatTensor(*data.shape).uniform_(-eps, eps)
        eta = eta.to(label.device)
        eta.requires_grad_()

        optimizer.zero_grad()
        LayerOneTrainner.param_optimizer.zero_grad()

        for j in range(K):
            pbar_dic = OrderedDict()
            TotalLoss = 0

            pred = net(data + eta.detach())

            loss = criterion(pred, label)
            TotalLoss = TotalLoss + loss
            wgrad = net.conv1.weight.grad
            TotalLoss.backward()
            net.conv1.weight.grad = wgrad


            p = -1.0 * net.layer_one_out.grad
            yofo_inp, eta = LayerOneTrainner.step(data, p, eta)

            with torch.no_grad():
                if j == 0:
                    acc = torch_accuracy(pred, label, (1,))
                    cleanacc = acc[0].item()
                    cleanloss = loss.item()

                if j == K - 1:
                    yofo_pred = net(yofo_inp)
                    yofoacc = torch_accuracy(yofo_pred, label, (1,))[0].item()

        optimizer.step()
        LayerOneTrainner.param_optimizer.step()
        optimizer.zero_grad()
        LayerOneTrainner.param_optimizer.zero_grad()
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['YoPoAcc'] = '{:.2f}'.format(yofoacc)
        pbar.set_postfix(pbar_dic)

    return cleanacc, yofoacc

if __name__ == "__main__":
    main()
