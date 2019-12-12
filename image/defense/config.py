import sys
import os
import argparse
import torch
from typing import Tuple, List, Dict


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(lib_dir)

class PieceWiseConstantLrSchedulerMaker(object):

    def __init__(self, milestones:List[int], gamma:float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)


class TrainingConfing():
    ## model saving path
    lib_dir = lib_dir

    ## training epcohs and batch size, evaluation intervals
    num_epochs = 105
    batch_size = 100
    val_interval = 10

    ## optimizer
    optimizer = 'SGD'
    learning_rate = 1e-4
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones=[70, 90, 100], gamma=0.1)
    momentum = 0.2
    weight_decay = 5e-4

config = TrainingConfing()



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=200, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--save_dir', default='', help = 'save your model to this path')
args = parser.parse_args()

if __name__ == '__main__':
    pass
