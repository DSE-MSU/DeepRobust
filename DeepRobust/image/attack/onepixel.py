import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()

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

def predict_classes(xs, img, target_calss, net, minimize=True):
	imgs_perturbed = perturb_image(xs, img.clone())
	input = Variable(imgs_perturbed, volatile=True).cuda()
	predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]

	return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):

	attack_image = perturb_image(x, img.clone())
	input = Variable(attack_image, volatile=True).cuda()
	confidence = F.softmax(net(input)).data.cpu().numpy()[0]
	predicted_class = np.argmax(confidence)

	if (verbose):
		print "Confidence: %.4f"%confidence[target_calss]
	if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
		return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
	# img: 1*3*W*H tensor
	# label: a number

	targeted_attack = target is not None
	target_calss = target if targeted_attack else label

	bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

	popmul = max(1, popsize/len(bounds))

	predict_fn = lambda xs: predict_classes(
		xs, img, target_calss, net, target is None)
	callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, net, targeted_attack, verbose)

	inits = np.zeros([popmul*len(bounds), len(bounds)])
	for init in inits:
		for i in range(pixels):
			init[i*5+0] = np.random.random()*32
			init[i*5+1] = np.random.random()*32
			init[i*5+2] = np.random.normal(128,127)
			init[i*5+3] = np.random.normal(128,127)
			init[i*5+4] = np.random.normal(128,127)

	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

	attack_image = perturb_image(attack_result.x, img)
	attack_var = Variable(attack_image, volatile=True).cuda()
	predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]

	predicted_class = np.argmax(predicted_probs)

	if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
		return 1, attack_result.x.astype(int)
	return 0, [None]




def main():
    
	results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
	print "Final success rate: %.4f"%results


if __name__ == '__main__':
	main()