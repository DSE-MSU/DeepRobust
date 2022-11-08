import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pickle

from deeprobust.image.attack.base_attack import BaseAttack

class FeatureScattering(BaseAttack):
    """
    FeatureScattering attack.
    """

    def __init__(self, model, device='cuda'):

        super(FeatureScattering, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate FeatureScattering adversarial examples.
        Parameters
        ----------
        image :
            original image
        label :
            target label
        kwargs :
            user defined paremeters
        """

        ## check and parse parameters for attack
        label = label.type(torch.FloatTensor)

        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return featurescattering_attack(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.num_steps,
                   self.step_size,
                   self.ls_factor,
                   self.clip_max,
                   self.clip_min,
                   self.train_flag,
                   self.print_process,
                   self.distance_measure,
                   self.device)

    def parse_params(self,
                     epsilon=0.031,
                     num_steps=10,
                     step_size=0.007,
                     ls_factor=0.1,
                     clip_max=1.0,
                     clip_min=0.0,
                     train_flag=True,
                     print_process=False,
                     distance_measure='l_inf'):
        """parse_params.
        Parameters
        ----------
        epsilon :
            perturbation constraint
        num_steps :
            iteration step
        step_size :
            step size
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        print_process :
            whether to print out the log during optimization process, True or False print out the log during optimization process, True or False.
        distance_measure:
            distance measurement used in adversarial example generation process. choice=['l_inf', 'l_2']
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.ls_factor = ls_factor
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.train_flag = train_flag
        self.print_process = print_process
        self.distance_measure = distance_measure
        
        return True

def featurescattering_attack(model,
                  x_natural,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  ls_factor,
                  clip_max,
                  clip_min,
                  print_process,
                  train_flag,
                  distance_measure,
                  device='cuda'):

    aux_net = pickle.loads(pickle.dumps(model))
    aux_net.eval()
    
    batch_size = len(x_natural)
    m = batch_size
    n = batch_size

    outputs = aux_net(x_natural)[0]
    targets_prob = F.softmax(outputs.float(), dim=1)
    y_tensor_adv = y
    # step_sign = 1.0

    # generate adversarial example
    x = x_natural.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)

    if train_flag:
        model.train()
    else:
        model.eval()

    logits_pred_nat, fea_nat = aux_net(x_natural)

    num_classes = logits_pred_nat.size(1)
    y_gt = one_hot_tensor(y, num_classes, device)

    loss_ce = softCrossEntropy()

    if distance_measure == 'l_inf':
        for i in range(num_steps):
            if print_process:
                print('generating at step ' + str(i + 1))

            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            ot_loss = sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat, logits_pred, None, None, 0.01, m, n, device)

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            logits_pred, fea = model(x)
            model.zero_grad()

            y_sm = label_smoothing(y_gt, y_gt.size(1), ls_factor)
            
            adv_loss = loss_ce(logits_pred, y_sm.detach())
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv, adv_loss

class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce

        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0).to(device)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0

    return y_tensor


def label_smoothing(y_batch_tensor, num_classes, delta):
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * y_batch_tensor + delta / (num_classes - 1)

    return y_batch_smooth

def sinkhorn_loss_joint_IPOT(alpha, beta, x_feature, y_feature, x_label, y_label, epsilon, m, n, device):
    C_fea = get_cost_matrix(x_feature, y_feature)
    C = C_fea
    T = sinkhorn(C, 0.01, 100, device)
    # T = IPOT(C, 1)
    batch_size = C.size(0)
    cost_ot = torch.sum(T * C)

    return cost_ot

def sinkhorn(C, epsilon, niter=50, device='cuda'):
    m = C.size(0)
    n = C.size(1)
    mu = Variable(1. / m * torch.FloatTensor(m).fill_(1).to(device),requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1).to(device),requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        """Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1

    def M(u, v):
        """Modified cost for logarithmic updates"""
        """$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"""
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        """log-sum-exp"""
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).cpu().data.numpy():
            break
    U, V = u, v

    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    pi = pi.to(device).float()

    return pi  # return the transport

def IPOT(cost_matrix, beta=1, device='cuda'):
    m = cost_matrix.size(0)
    n = cost_matrix.size(1)
    sigma = 1.0 / n * torch.ones([n, 1]).to(device)

    T = torch.ones([m, n]).to(device)
    A = torch.exp(-cost_matrix / beta)

    for t in range(50):
        # BUG: should be elementwise product, * in numpy
        #Q = torch.mm(A, T)
        Q = A * T  # Hardmard product
        for k in range(1):
            delta = 1.0 / (m * torch.mm(Q, sigma))
            sigma = 1.0 / (n * torch.mm(delta.t(), Q)).t()
            #sigma = 1.0 / (n * torch.mv(Q, delta))
        tmp = torch.mm(construct_diag(torch.squeeze(delta)), Q)
        T = torch.mm(tmp, construct_diag(torch.squeeze(sigma)))

    return T

def construct_diag(d):
    n = d.size(0)
    x = torch.zeros([n, n]).to(d.device)
    x[range(n), range(n)] = d.view(-1)

    return x

def get_cost_matrix(x_feature, y_feature):
    C_fea = cost_matrix_cos(x_feature, y_feature)  # Wasserstein cost function

    return C_fea

def cost_matrix_cos(x, y, p=2):
    # return the m*n sized cost matrix
    """Returns the matrix of $|x_i-y_j|^p$."""
    # un squeeze differently so that the tensors can broadcast
    # dim-2 (summed over) is the feature dim
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    c = torch.clamp(1 - cos(x_col, y_lin), min=0)

    return c

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

    