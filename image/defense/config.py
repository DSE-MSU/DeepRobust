from easydict import EasyDict
import sys
import os
import argparse
import numpy as np
import torch

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(lib_dir)

from training.config import TrainingConfigBase, SGDOptimizerMaker, \
    PieceWiseConstantLrSchedulerMaker, IPGDAttackMethodMaker

class TrainingConfing(TrainingConfigBase):

    lib_dir = lib_dir

    num_epochs = 105
    val_interval = 10
    weight_decay = 5e-4

    inner_iters = 5
    K = 2
    sigma = 0.007
    eps = 0.031

    create_optimizer = SGDOptimizerMaker(lr =2e-1, momentum = 0.9, weight_decay = weight_decay)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [70, 90, 100], gamma = 0.1)

    create_loss_function = torch.nn.CrossEntropyLoss

    create_attack_method = None

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))


config = TrainingConfing()



parser = argparse.ArgumentParser()

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


if __name__ == '__main__':
    pass