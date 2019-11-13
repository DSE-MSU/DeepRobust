import pytorch as torch
from torch import optim

import attack.base_attack as base

def to_attack_space(x):
    # map from [min_, max_] to [-1, +1]
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999

    # from (-1, +1) to (-inf, +inf)
    return np.arctanh(x)

def to_model_space(x):
    """Transforms an input from the attack space
    to the model space. This transformation and
    the returned gradient are elementwise."""

    # from (-inf, +inf) to (-1, +1)
    x = np.tanh(x)

    grad = 1 - np.square(x)

    # map from (-1, +1) to (min_, max_)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a

    grad = grad * b
    return x, grad

class CarliniWagner(base.BaseAttack):

    def __init__(self, model, device = 'cuda'):
        super()
        self.model = model
        self.device = device
    
    def generate(self, image, label, **kwargs):

        assert self.parse_params(**kwargs)
        return cw(self.model, self.image, self.label, )

    def parse_params(confidence, clip_max, clip_min, max_iterations, initial_const, binary_search_steps, learning_rate, abort_early):
        self.epsilon = epsilon
        self.ord_ = ord_
        self.T = T
        self.alpha = alpha
        self.clip = clip
        return True

    def to_attack_space(x):
        # map from [min_, max_] to [-1, +1]
        # x'=(x- 0.5 * (max+min) / 0.5 * (max-min))
        a = (self.clip_min + self.clip_max) / 2
        b = (self.clip_max - self.clip_min) / 2
        x = (x - a) / b

        # from [-1, +1] to approx. (-1, +1)
        x = x * 0.999999

        # from (-1, +1) to (-inf, +inf)
        return np.arctanh(x)

    def to_model_space(x):
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

    def cw(self, model, image, label, confidence, clip_max, clip_min, max_iterations, initial_const, binary_search_steps, learning_rate):
        
        #change the input image
        img_tanh = to_attack_space(image)
        img_ori ,_ = to_model_space(img_tanh)

        #do binary search
        c = initial_const
        c_low = 0
        c_high = np.inf

        for step in range(max_iterations):
            print("starting optimization.")

            #initialize perturbation
            perturbation = np.zeros_like(img_tanh)

            optimizer = AdamOptimizer(advexample.shape)
            
            is_adversarial = False
            loss = np.inf

            for iteration in range(max_iterations):
                img_adv, adv_grid = to_model_space(img_tanh + perturbation)
                logits = model.
                loss, loss_grad = self.loss_function(
                    c, a, img_adv, logits, img_ori, confidence, clip_min, clip_max 
                )
                
                gradient = adv_grid * loss_grid
                perturbation += optimizer(gradient, learning_rate)

                if is_adversarial:
                    found_adv = True
            
            #do binary search on c
            if found_adv:
                c_high = c
            else:
                c_low = c

            if upper_bound == np.inf:
                c *= 10
            else:
                c = (c_high + c_low) / 2

    @classmethod
    def loss_function(
        cls, const, a, x, logits, reconstructed_original, confidence, min_, max_
    ):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        targeted = a.target_class is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class)
            c_maximize = a.target_class
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_l2_distance = np.sum((x - reconstructed_original) ** 2) / s ** 2
        total_loss = squared_l2_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = yield from a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s ** 2) * (x - reconstructed_original)

        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)





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
    
    