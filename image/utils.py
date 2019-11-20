import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os

def create_train_dataset(batch_size = 128, root = '../data'):
    """
    Create different training dataset
    """

    transform_train = transforms.Compose([
    transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader

def create_test_dataset(batch_size = 128, root = '../data'):
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader

def save_checkpoint(now_epoch, net, optimizer, lr_scheduler, file_name):
    checkpoint = {'epoch': now_epoch,
                  'state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler_state_dict':lr_scheduler.state_dict()}
    if os.path.exists(file_name):
        print('Overwriting {}'.format(file_name))
    torch.save(checkpoint, file_name)
    # link_name = os.path.join(*file_name.split(os.path.sep)[:-1], 'last.checkpoint')
    # #print(link_name)
    # make_symlink(source = file_name, link_name=link_name)

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
    """
    Note: overwriting enabled!
    """

    if os.path.exists(link_name):
        print("Link name already exist! Removing '{}' and overwriting".format(link_name))
        os.remove(link_name)
    if os.path.exists(source):
        os.symlink(source, link_name)
        return
    else:
        print('Source path not exists')    

def onehot_like(a, index, value=1):
    """Creates an array like a, with all values
    set to 0 except one.
    Parameters
    ----------
    a : array_like
        The returned one-hot array will have the same shape
        and dtype as this array
    index : int
        The index that should be set to `value`
    value : single value compatible with a.dtype
        The value to set at the given index
    Returns
    -------
    `numpy.ndarray`
        One-hot array with the given value at the given
        location and zeros everywhere else.
    """
    #TODO: change the note here.
    x = np.zeros_like(a)
    x[index] = value
    return x

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

def arctanh(x, eps=1e-6):
    """
    Calculate arctanh(x)
    """
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5

def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = torch.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x*x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)