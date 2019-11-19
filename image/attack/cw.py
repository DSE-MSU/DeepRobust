import torch
from torch import optim
import numpy as np
import logging

from DeepRobust.image.attack.base_attack import BaseAttack
from DeepRobust.image.utils import onehot_like

class CarliniWagner(BaseAttack):

    def __init__(self, model, device = 'cuda'):
        super(CarliniWagner, self).__init__(model, device)
        self.model = model
        self.device = device
    
    def generate(self, image, label, **kwargs):
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        return self.cw(self.model, 
                  self.image, 
                  self.label,
                  self.target,
                  self.confidence,
                  self.clip_min,
                  self.clip_min,
                  self.max_iterations,
                  self.initial_const,
                  self.binary_search_steps,
                  self.learning_rate
                  )

    def parse_params(self,
                     target,
                     classnum,
                     confidence, 
                     clip_max, 
                     clip_min, 
                     max_iterations, 
                     initial_const, 
                     binary_search_steps, 
                     learning_rate, 
                     abort_early):

        self.target = target
        self.classnum = classnum
        self.confidence = confidence
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.abort_early = abort_early
        return True

    def to_attack_space(self, x):
        # map from [min_, max_] to [-1, +1]
        # x'=(x- 0.5 * (max+min) / 0.5 * (max-min))
        a = (self.clip_min + self.clip_max) / 2
        b = (self.clip_max - self.clip_min) / 2
        x = (x - a) / b

        # from [-1, +1] to approx. (-1, +1)
        x = x * 0.999999

        # from (-1, +1) to (-inf, +inf)
        return np.arctanh(x)

    def to_model_space(self, x):
        """Transforms an input from the attack space
        to the model space. This transformation and
        the returned gradient are elementwise."""

        # from (-inf, +inf) to (-1, +1)
        x = np.tanh(x)

        grad = 1 - np.square(x)

        # map from (-1, +1) to (min_, max_)
        a = (self.clip_min + self.clip_max) / 2
        b = (self.clip_max - self.clip_min) / 2
        x = x * b + a

        grad = grad * b
        return x, grad

    def cw(self, model, image, label, target, confidence, clip_max, clip_min, max_iterations, initial_const, binary_search_steps, learning_rate):
        
        #change the input image
        img_tanh = self.to_attack_space(image.cpu())
        img_ori ,_ = self.to_model_space(img_tanh)
        img_ori = img_ori.to(self.device)

        #binary search initialization
        c = initial_const
        c_low = 0
        c_high = np.inf
        found_adv = False
        last_loss = np.inf

        for step in range(max_iterations):

            #initialize perturbation
            perturbation = torch.from_numpy(np.zeros_like(img_tanh))

            optimizer = AdamOptimizer(img_tanh.shape)
            
            is_adversarial = False

            for iteration in range(max_iterations):

                # adversary example
                img_adv, adv_grid = self.to_model_space(img_tanh + perturbation)
                img_adv = img_adv.to(self.device)
                img_adv.requires_grad = True
                
                #output of the layer before softmax
                output = model.get_logits(img_adv)

                #pending success
                is_adversarial = self.pending_f(img_adv)

                #calculate loss function and gradient of loss funcition on x
                loss, loss_grad = self.loss_function(
                    img_adv, c, self.target, img_ori, self.confidence, self.clip_min, self.clip_max 
                )
                
                #calculate gradient of loss function on w
                gradient = adv_grid.to(self.device) * loss_grad.to(self.device)
                perturbation = perturbation + torch.from_numpy(optimizer(gradient.cpu().detach().numpy(), learning_rate)).float()
                if is_adversarial:
                    found_adv = True
            
            #do binary search on c
            if found_adv:
                c_high = c
            else:
                c_low = c

            if c_high == np.inf:
                c *= 10
            else:
                c = (c_high + c_low) / 2
        
            print("optimization itaration:{:.1f},loss:{:.4f}".format(step,loss))
    
            #abort early
            if(self.abort_early == True and (step % 10) == 0 and step > 15) :
                print(loss, last_loss)
                if not (loss <= 0.999 * last_loss):
                    break
                last_loss = loss
        
        return img_adv

    def loss_function(
        self, x_p, const, target, reconstructed_original, confidence, min_, max_
    ):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        ## get the output of model before softmax
        x_p.requires_grad = True
        logits = self.model.get_logits(x_p).to(self.device)

        ## find the largest class except the target class
        targetlabel_mask = (torch.from_numpy(onehot_like(np.zeros(self.classnum), target))).double()
        secondlargest_mask = (torch.from_numpy(np.ones(self.classnum)) - targetlabel_mask).to(self.device)
        
        secondlargest = np.argmax((logits.double() * secondlargest_mask).cpu().detach().numpy())

        # print(logits)
        # print(secondlargest)
        # print(target)
        # print(targetlabel_mask)
        # print(secondlargest_mask)
        # print((logits.double() * secondlargest_mask).cpu().detach().numpy())
        
        is_adv_loss = logits[0][secondlargest] - logits[0][target]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence

        if is_adv_loss == 0:
            is_adv_loss_grad = 0
        else:
            is_adv_loss.backward()
            is_adv_loss_grad = x_p.grad

        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_l2_distance = np.sum( ((x_p - reconstructed_original) ** 2).cpu().detach().numpy() ) / s ** 2
        total_loss = squared_l2_distance + const * is_adv_loss


        squared_l2_distance_grad = (2 / s ** 2) * (x_p - reconstructed_original)
        
        #print(is_adv_loss_grad)
        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad
    
    def pending_f(self, x_p):
        """Pending is the loss function is less than 0
        """
        targetlabel_mask = torch.from_numpy(onehot_like(np.zeros(self.classnum), self.target))
        secondlargest_mask = torch.from_numpy(np.ones(self.classnum)) - targetlabel_mask
        targetlabel_mask = targetlabel_mask.to(self.device)
        secondlargest_mask = secondlargest_mask.to(self.device)
        
        Zx_i = np.max((self.model.get_logits(x_p).double().to(self.device) * secondlargest_mask).cpu().detach().numpy())
        Zx_t = np.max((self.model.get_logits(x_p).double().to(self.device) * targetlabel_mask).cpu().detach().numpy())

        if ( Zx_i - Zx_t  < - self.confidence):
            return True
        else:
            return False


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable. 
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """
    #TODO Add reference or rewrite the function.
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """

        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    