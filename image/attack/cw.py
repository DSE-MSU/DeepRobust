import pytorch as torch

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
        self.model = model
        self.device = device
    
    def generate(self, image, label, **kwargs):

        assert self.parse_params(**kwargs)
        return cw(self.model, image, label)

    def parse_params(epsilon = 1.0, ord_ = 2, T = 2, alpha = 0.9, min_prob = 0, clip):
        self.epsilon = epsilon
        self.ord_ = ord_
        self.T = T
        self.alpha = alpha
        self.clip = clip
        return True

    def cw(self, )