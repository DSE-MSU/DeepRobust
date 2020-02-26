"""
This is a re-implementation of One pixel attack.

Reference:
Akhtar, N., & Mian, A. (2018).
Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey: A Survey.
IEEE Access, 6, 14410-14430.

Reference code: https://github.com/DebangLi/one-pixel-attack-pytorch
"""

import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from deeprobust.image.optimizer import differential_evolution
from deeprobust.image.attack.base_attack import BaseAttack
from deeprobust.image.utils import progress_bar

class Onepixel(BaseAttack):

	def __init__(self, model, device = 'cuda'):

		super(Onepixel, self).__init__(model, device)

	def generate(self, image, label, **kwargs):
		label = label.type(torch.FloatTensor)

        ## check and parse parameters for attack
		assert self.check_type_device(image, label)
		assert self.parse_params(**kwargs)

		return self.one_pixel(self.image,
						 self.label,
						 self.targeted_attack,
						 self.pixels,
						 self.maxiter,
						 self.popsize,
						 self.print_log)

	def get_pred():
		return self.adv_pred
	
	def parse_params(self,
					pixels = 1,
					maxiter = 100,
					popsize = 400,
					samples = 100,
					targeted_attack = False,
					print_log = True,
					target = 0):
		self.pixels = pixels
		self.maxiter = maxiter
		self.popsize = popsize
		self.samples = samples
		self.targeted_attack = targeted_attack
		self.print_log = print_log
		self.target = target
		return True


	def one_pixel(self, img, label, targeted_attack = False, target = 0, pixels = 1, maxiter = 75, popsize = 400, print_log = False):
		# img: 1*3*W*H tensor
		# label: a number

		target_calss = target if targeted_attack else label

		bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

		popmul = max(1, popsize/len(bounds))

		predict_fn = lambda xs: predict_classes(
			xs, img, target_calss, self.model, targeted_attack, self.device)
		callback_fn = lambda x, convergence: attack_success(
			x, img, target_calss, self.model, targeted_attack, print_log, self.device)

		inits = np.zeros([popmul*len(bounds), len(bounds)])
		for init in inits:
			for i in range(pixels):
				init[i*5+0] = np.random.random()*32
				init[i*5+1] = np.random.random()*32
				init[i*5+2] = np.random.normal(128,127)
				init[i*5+3] = np.random.normal(128,127)
				init[i*5+4] = np.random.normal(128,127)

		attack_result = differential_evolution(predict_fn, bounds, maxiter = maxiter, popsize = popmul,
			recombination = 1, atol = -1, callback = callback_fn, polish = False, init = inits)

		attack_image = perturb_image(attack_result.x, img)
		attack_var = Variable(attack_image, volatile=True).cuda()
		predicted_probs = F.softmax(self.model(attack_var)).data.cpu().numpy()[0]

		predicted_class = np.argmax(predicted_probs)

		if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
			self.adv_pred = predict_class
			return attack_image
		return [None]

def perturb_image(xs, img):

	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)

	count = 0

	for x in xs:
		pixels = np.split(x, len(x)/5)

		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
		count += 1

	return imgs

def predict_classes(xs, img, target_calss, net, minimize=True, device = 'cuda'):
	imgs_perturbed = perturb_image(xs, img.clone()).to(device)
	predictions = F.softmax(net(imgs_perturbed)).data.cpu().numpy()[:, target_calss]

	return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack = False, print_log=False, device = 'cuda'):

	attack_image = perturb_image(x, img.clone()).to(device)
	confidence = F.softmax(net(attack_image)).data.cpu().numpy()[0]
	pred = np.argmax(confidence)

	if (print_log):
		print("Confidence: %.4f"%confidence[target_calss])
	if (targeted_attack and pred == target_calss) or (not targeted_attack and pred != target_calss):
		return True





