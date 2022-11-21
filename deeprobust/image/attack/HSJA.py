import numpy as np
from typing import Callable, Optional
from resnet import *
import torch
from torchvision import datasets, transforms
def binsearch_boundary(src_pt,
                      dest_pt,
                      threshold,
                      model, target_label
                     ) -> np.array:
    '''
    Find a point between two points that will lies on the boundary.
    :param src_pt:    point at which phi=0
    :param dest_pt:   point at which phi=1
    :param threshold: gap between source point and destination point
    '''

    while torch.linalg.norm(dest_pt - src_pt) >= threshold:
        midpoint = (src_pt + dest_pt) / 2
        if indicator_function(model, midpoint, target_label) == 1:
            dest_pt = midpoint
        else:
            src_pt = midpoint
    return dest_pt

def indicator_function(model, x, target_label):
    if torch.argmax(model(x)) == target_label:
        return 1
    else:
        return 0

def estimate_gradient(orig_pt,
                      step_size,
                      sample_count,
                      model, target_label
                     ) -> np.ndarray:
    '''
    Estimate the gradient via Monte Carlo sampling.
    :param orig_pt:      point to estimate gradient at
    :param step_size:    length of each step in the proposed direction
    :param sample_count: number of Monte Carlo samples
    '''
    # sample directions

    directions = torch.from_numpy(np.random.randn(sample_count, orig_pt.shape[1], orig_pt.shape[2], orig_pt.shape[3])).float().to('cuda')
    directions /= torch.linalg.norm(directions, dim = (1, 2, 3), dtype = torch.float).reshape(sample_count, 1, 1, 1).repeat(1, orig_pt.shape[1], orig_pt.shape[2], orig_pt.shape[3])

    # get phi values
    values = torch.from_numpy(np.empty((sample_count, 1), dtype = np.float)).to('cuda')

    for i in range(sample_count):
        values[i, 0] = indicator_function(model, orig_pt + directions[i, :] * step_size, target_label) * 2 - 1
    # subtract from the mean

    avg = torch.sum(directions * values.reshape(sample_count, 1, 1 ,1), dim = 0) / (sample_count - 1)
    # project them to unit L2
    norm_avg = avg / torch.linalg.norm(avg)
    return norm_avg.float()

def gradient_descent(orig_pt,
                     grad,
                     step_size,
                     model, target_label
                    ) -> np.ndarray:
    '''
    Do gradient descent on a point already on the boundary.
    :param orig_pt:    point to do gradient descent on
    :param grad:       the estimated gradient
    :param step_size:  initial step size to try
    '''
    # find the step size to stay in phi=1

    while True:
        new_vector = orig_pt + step_size * grad
        if indicator_function(model, new_vector, target_label):
            break
        step_size /= 2
    return new_vector

def hopskipjumpattack(orig_pt,
                      model,
                      max_iter: Optional[int] = 100,
                      init_grad_queries: Optional[int] = 100,
                      binsearch_threshold: Optional[float] = 2e-6,
                      dest_pt = None,
                      target_label = None,
                      device = 'cuda'
                     ) -> torch.Tensor():
    '''
    Implementation of the HopSkipJumpAttack.
    :param orig_pt:             point at which phi=0
    :param max_iter:            (Optional) maximum number of optimization iteration.
                                Default is 100.
    :param init_grad_queries:   (Optional) initial query count to estimate gradient
                                Default is 100.
    :param binsearch_threshold: (Optional) the threshold to stop binary searching the boundary.
                                Default is 1e-6.
    :param dest_pt:             (Optional) point which phi=1.
                                If dest_pt is None, will be initialized to be a random vector.
                                For cases when one restarts this iterative algo. Default is 0.
    '''
    d = orig_pt.shape[0]
    # initialize a vector with phi=1
    if dest_pt is None:
        while True:
            dest_pt = torch.from_numpy(np.random.random_sample(d)).to(device)
            if indicator_function(model, dest_pt, target_label) == 1:
                break

    for it in range(1, max_iter + 1):

        print(f'Iter {it:03d}: ', end='')

        # project on to boundary
        boundary = binsearch_boundary(orig_pt, dest_pt, binsearch_threshold, model, target_label)
        # if the error is too small, return as is
        distance = torch.linalg.norm(boundary - orig_pt)
        if distance < binsearch_threshold:
            print('Step size too small, terminating...')
            # this works because we return the phi=1 endpoint in binsearch.
            return boundary

        # estimate the gradient
        step_size = torch.linalg.norm(dest_pt - orig_pt) / d
        sample_count = int(init_grad_queries * it ** 0.5)
        grad = estimate_gradient(boundary, step_size, sample_count, model, target_label)

        # and gradient descend
        step_size = torch.linalg.norm(boundary - orig_pt) / it ** 0.5
        dest_pt = gradient_descent(boundary, grad, step_size, model, target_label)
        distance = torch.linalg.norm(dest_pt - orig_pt)
        print(distance)

    return dest_pt

if __name__ == '__main__':

    model = ResNet18().to('cuda')
    state_dict = torch.load("./clean.pt")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if (k[0:7] == 'module.'):
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


    dataset = datasets.CIFAR10(root = '~/Documents/data/', train = False, transform = transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    ori_img, ori_label = next(iter(testloader))
    iterator = iter(testloader)
    target_img, target_label = next(iterator)

    ori_img, ori_label = ori_img.to('cuda'), ori_label.to('cuda')
    target_img, target_label = target_img.to('cuda'), target_label.to('cuda')

    while (ori_label == target_label or torch.argmax(model(target_img)) != target_label):
        target_img, target_label = next(iterator)
        ori_img, ori_label = ori_img.to('cuda'), ori_label.to('cuda')
        target_img, target_label = target_img.to('cuda'), target_label.to('cuda')


    print('original label:', ori_label)
    print('target label:', target_label)
    print(torch.argmax(model(target_img)) == target_label)
    print(indicator_function(model, target_img, target_label))
    adv_img = hopskipjumpattack(ori_img, model, dest_pt = target_img, target_label = target_label)
    print('attack label:', torch.argmax(model(adv_img)))
