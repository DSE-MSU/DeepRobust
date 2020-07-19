import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import urllib.request

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

def download_model(url, file):
    print('Dowloading from {} to {}'.format(url, file))
    try:
        urllib.request.urlretrieve(url, file)
    except:
        raise Exception("Download failed! Make sure you have stable Internet connection and enter the right name")

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

from texttable import Texttable
def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    input:
        param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

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

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """decrease the learning rate"""
    lr = learning_rate
    if epoch >= 55:
        lr = learning_rate * 0.1
    if epoch >= 75:
        lr = learning_rate * 0.01
    if epoch >= 90:
        lr = learning_rate * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

