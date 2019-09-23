import torch
import torch.nn as nn
import scipy.optimize as so
import numpy as np

from attack import base_attack
 
class LBFGS(base_attack):

    def __init__(self, model, target_label, device = 'cuda' ):  
        super(LBFGS, self).__init__(model, device)
        self.target_label = target_label

    def generate(self, input, label, **kwargs):
        assert self.check_type_device(input, label)
        assert self.parse_params(**kwargs)
        return optimize(self.model, 
                        self.input, 
                        self.label, 
                        self.target_label, 
                        self.bounds, 
                        self.epsilon, 
                        self.maxiter, 
                        self.class_num)

    def parse_params(self,
                     clip_max,
                     clip_min,
                     class_num,
                     epsilon = 1e-5,
                     maxiter = 150,
                     ):
        self.epsilon = epsilon
        self.maxiter = maxiter 
        self.class_num = class_num
        self.bounds = (clip_min, clip_max)
        return True

def optimize(model, input, label, target_label, bounds, epsilon, maxiter, class_num):
        x0 = input.numpy()
        min_, max_ = bounds
        target_dist = torch.zeros(class_num)
        target_dist[target_label] = 1        

        # store the shape for later and operate on the flattened input
        shape = x0.shape
        dtype = x0.dtype
        x0 = x0.flatten().astype(np.float64)

        n = len(x0)
        bounds = [(min_, max_)] * n

        def distance(x,y):
            # calculate the distance 
            dist_squ = torch.norm(x - y)
            return dist_squ **2

        def loss(x, c):
            x = x.astype(dtype)
            v1 = distance(x0,x)
            loss = nn.CrossEntropyLoss()
            v2 = loss(x0, target_dist)
            v = v1 + c * v2
            return np.float64(v)

        def lbfgs_b(c):
            approx_grad_eps = (max_ - min_) / 100
            x, f, d = so.fmin_l_bfgs_b(
                loss,
                x0,
                args=(c,),
                approx_grad = True,
                bounds = bounds,
                m = 15,
                maxiter = maxiter,
                epsilon = approx_grad_eps)

            logging.info(d)

            # LBFGS-B does not always exactly respect the boundaries
            if np.amax(x) > max_ or np.amin(x) < min_:   # pragma: no coverage
                logging.info('Input out of bounds (min, max = {}, {}). Performing manual clip.'.format(
                    np.amin(x), np.amax(x)))
                x = np.clip(x, min_, max_)

            is_adversarial = pend_attack(trainedmodel, x.reshape(shape).astype(dtype), target_label)
            return is_adversarial

        # finding initial c
        c = epsilon
        for i in range(30):
            c = 2 * c
            is_adversarial = lbfgs_b(c)
            logging.info('Tested c = {:.4e}: {}'.format(
                c,
                ('adversarial' if is_adversarial else 'not adversarial')))
            if is_adversarial:
                break
        else:  # pragma: no cover
            logging.info('Could not find an adversarial; maybe the model returns wrong gradients')
            return

        # binary search
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            c_half = (c_low + c_high) / 2
            is_adversarial = lbfgs_b(c_half)
            logging.info('Tested c = {:.4e}: {} ({:.4e}, {:.4e})'.format(
                c_half,
                ('adversarial' if is_adversarial else 'not adversarial'),
                c_low,
                c_high))
            if is_adversarial:
                c_high = c_half
            else:
                c_low = c_half

def pending_attack(self, target_model, adv_exp, target_label):
    predict1 = target_model(adv_exp)
    label = predict1.argmax(dim=1, keepdim=True)
    if result == target:
        return True
    else:
        return False


