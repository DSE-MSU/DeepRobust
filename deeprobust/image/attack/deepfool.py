import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

from deeprobust.image.attack.base_attack import BaseAttack

class DeepFool(BaseAttack):
    """DeepFool attack.
    """

    def __init__(self, model, device = 'cuda' ):
        super(DeepFool, self).__init__(model, device)
        self.model = model
        self.device = device

    def generate(self, image, label, **kwargs):
        """
        Call this function to generate adversarial examples.

        Parameters
        ----------
        image : 1*H*W*3
            original image
        label : int
            target label
        kwargs :
            user defined paremeters
       
        Returns
        -------
        adv_img :
            adversarial examples
        """
 

        #check type device
        assert self.check_type_device(image, label)
        is_cuda = torch.cuda.is_available()

        if (is_cuda and self.device == 'cuda'):
            self.image = image.cuda()
            self.model = self.model.cuda()
        else:
            self.image = image

        assert self.parse_params(**kwargs)

        adv_img, self.r, self.ite =  deepfool(self.model,
                                  self.image,
                                  self.num_classes,
                                  self.overshoot,
                                  self.max_iteration,
                                  self.device)
        return adv_img

    def getpert(self):
        return self.r, self.ite

    def parse_params(self,
                     num_classes = 10,
                     overshoot = 0.02,
                     max_iteration = 50):
        """
        Parse the user defined parameters

        Parameters
        ----------
        num_classes : int
            limits the number of classes to test against. (default = 10)
        overshoot : float
            used as a termination criterion to prevent vanishing updates (default = 0.02).
        max_iteration : int
            maximum number of iteration for deepfool (default = 50)
        """
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iteration = max_iteration
        return True

def deepfool(model, image, num_classes, overshoot, max_iter, device):
    f_image = model.forward(image).data.cpu().numpy().flatten()
    output = (np.array(f_image)).flatten().argsort()[::-1]

    output = output[0:num_classes]
    label = output[0]

    input_shape = image.cpu().numpy().shape
    x = copy.deepcopy(image).requires_grad_(True)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    fs = model.forward(x)
    fs_list = [fs[0,output[k]] for k in range(num_classes)]
    current_pred_label = label

    for i in range(max_iter):

        pert = np.inf
        fs[0, output[0]].backward(retain_graph = True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, output[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, output[k]] - fs[0, output[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

        x = pert_image.detach().requires_grad_(True)
        fs = model.forward(x)

        if (not np.argmax(fs.data.cpu().numpy().flatten()) == label):
            break


    r_tot = (1+overshoot)*r_tot

    return pert_image, r_tot, i

