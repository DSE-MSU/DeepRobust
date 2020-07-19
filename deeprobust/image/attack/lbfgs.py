import torch
import torch.nn as nn
import scipy.optimize as so
import numpy as np
import torch.nn.functional as F #233

from deeprobust.image.attack.base_attack import BaseAttack

class LBFGS(BaseAttack):
    """
    LBFGS is the first adversarial generating algorithm.    
    """


    def __init__(self, model, label, device = 'cuda' ):
        super(LBFGS, self).__init__(model, device)

    def generate(self, image, label, target_label, **kwargs):
        """
        Call this function to generate adversarial examples.

        Parameters
        ----------
        image :
            original image
        label :
            target label
        kwargs :
            user defined paremeters
        """
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        self.target_label = target_label
        adv_img, self.dist, self.loss = optimize(self.model,
                                       self.image,
                                       self.label,
                                       self.target_label,
                                       self.bounds,
                                       self.epsilon,
                                       self.maxiter,
                                       self.class_num,
                                       self.device)
        return adv_img

    def distance(self):
        return self.dist

    def loss(self):
        return self.loss

    def parse_params(self,
                     clip_max = 1,
                     clip_min = 0,
                     class_num = 10,
                     epsilon = 1e-5,  #step of finding initial c
                     maxiter = 20,    #maximum of iteration in lbfgs optimization
                     ):
        """
        Parse the user defined parameters.

        Parameters
        ----------
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        class_num :
            total number of class
        epsilon :
            step length for binary seach
        maxiter :
            maximum number of iterations
        """
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.class_num = class_num
        self.bounds = (clip_min, clip_max)
        return True

def optimize(model, image, label, target_label, bounds, epsilon, maxiter, class_num, device):
    x_t = image
    x0 = image[0].to('cpu').detach().numpy()
    min_, max_ = bounds

    target_dist = torch.tensor(target_label)
    target_dist = target_dist.unsqueeze_(0).long().to(device)

    # store the shape for later and operate on the flattened input

    shape = x0.shape
    dtype = x0.dtype
    x0 = x0.flatten().astype(np.float64)

    n = len(x0)
    bounds = [(min_, max_)] * n

    def distance(x,y):
        # calculate the distance
        x = torch.from_numpy(x).double()
        y = torch.from_numpy(y).double()

        dist_squ = torch.norm(x - y)
        return dist_squ **2

    def loss(x, c):
        #calculate the target function
        v1 = distance(x0,x)

        x = torch.tensor(x.astype(dtype).reshape(shape))
        x = x.unsqueeze_(0).float().to(device)

        predict = model(x)
        v2 = F.nll_loss(predict, target_dist)

        v = c * v1 + v2
        #print(v)
        return np.float64(v)

    def pending_attack(target_model, adv_exp, target_label):
        # pending if the attack success
        adv_exp = adv_exp.reshape(shape).astype(dtype)
        adv_exp = torch.from_numpy(adv_exp)
        adv_exp = adv_exp.unsqueeze_(0).float().to(device)

        predict1 = target_model(adv_exp)
        label = predict1.argmax(dim=1, keepdim=True)
        if label == target_label:
            return True
        else:
            return False

    def lbfgs_b(c):

        #initial the variables
        approx_grad_eps = (max_ - min_) / 100
        print('in lbfgs_b:', 'c =', c)

        #start optimization
        optimize_output, f, d = so.fmin_l_bfgs_b(
                loss,
                x0,
                args=(c,),
                approx_grad = True,
                bounds = bounds,
                m = 15,
                maxiter = maxiter,
                factr = 1e10,  #optimization accuracy
                maxls = 5,
                epsilon = approx_grad_eps)
        print('finish optimization')

        # LBFGS-B does not always exactly respect the boundaries
        if np.amax(optimize_output) > max_ or np.amin(optimize_output) < min_:   # pragma: no coverage
            logging.info('Input out of bounds (min, max = {}, {}). Performing manual clip.'.format(
                    np.amin(optimize_output), np.amax(optimize_output)))

            optimize_output = np.clip(optimize_output, min_, max_)

        #optimize_output = optimize_output.reshape(shape).astype(dtype)
        #test_input = torch.from_numpy(optimize_output)
        #print(test_input)
        #test_input = test_input.unsqueeze_(0).float()
        is_adversarial = pending_attack(target_model = model, adv_exp = optimize_output, target_label = target_label)
        return optimize_output, is_adversarial

    #x_new, isadv = lbfgs_b(0)


    # finding initial c
    c = epsilon
    print('finding initial c:')

    for i in range(30):
        c = 2 * c
        x_new, is_adversarial = lbfgs_b(c)
        if is_adversarial == False:
            break

    print('start binary search:')
    if is_adversarial == True:  # pragma: no cover
        print('Could not find an adversarial; maybe the model returns wrong gradients')
        return

    print('c_high:',c)

    # binary search
    c_low = 0
    c_high = c
    while c_high - c_low >= epsilon:
        print(c_high,' ',c_low)
        c_half = (c_low + c_high) / 2
        x_new, is_adversarial = lbfgs_b(c_half)

        if is_adversarial:
            c_low = c_half
        else:
            c_high = c_half

    x_new, is_adversarial = lbfgs_b(c_low)
    dis = distance(x_new, x0)
    mintargetfunc = loss(x_new, c_low)

    x_new = x_new.astype(dtype)
    x_new = x_new.reshape(shape)

    x_new = torch.from_numpy(x_new).unsqueeze_(0).float().to(device)

    return x_new, dis, mintargetfunc



