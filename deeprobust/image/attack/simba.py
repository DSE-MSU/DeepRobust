import torchvision.datasets as dset
import torchvision.transforms as trans
import math
import random
import argparse
import os
import sys
sys.path.append('pytorch-cifar')
from resnet import *

import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFROM = trans.Compose([trans.Resize(256),
         trans.CenterCrop(224),
         trans.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_TRANSFROM = trans.Compose([
    trans.Resize(342),
    trans.CenterCrop(299),
    trans.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = trans.Compose([trans.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFROM = trans.Compose([trans.ToTensor()])

# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


# get most likely predictions and probabilities for a set of inputs
def get_preds(model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)

    return all_preds, all_probs

class SimBA:

    def __init__(self, model, dataset, image_size):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z

    def normalize(self, x):
        return apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        output = self.model(self.normalize(x))
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)

    def get_preds(self, x):
        output = self.model(self.normalize(x))
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False, device = 'cuda'):

        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0).to(device)
        y = y.to(device)
        last_prob = self.get_probs(x, y)
        print('correct label:', y)
        print('predict label:', self.get_preds(x))

        for i in range(num_iters):
            diff = torch.zeros(n_dims).to('cuda')
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)

            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + epsilon * diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            #if i % 10 == 0:
            print(last_prob)
            if self.get_preds(x) != y:
                print('wrong label:', self.get_preds(x))
                return x.squeeze()
        return x.squeeze()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')

    parser.add_argument('--result_dir', type=str, default='save_cifar', help='directory for saving results')
    parser.add_argument('--sampled_image_dir', type=str, default='save_cifar', help='directory to cache sampled images')
    parser.add_argument('--model', type=str, default='resnet18', help='type of base model to use')
    #parser.add_argument('--model_ckpt', type=str, required=True, help='model checkpoint location')
    parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
    parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
    parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
    parser.add_argument('--freq_dims', type=int, default=32, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, default=7, help='stride for block order')
    parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
    args = parser.parse_args()

    if not os.path.exists(args.sampled_image_dir):
        os.mkdir(args.sampled_image_dir)
    current_path = "./"
    data_root = os.path.join(current_path, 'data/')
    model_ckpt = './clean.pt'
    # load model and dataset
    model = ResNet18().cuda()
    #model = torch.nn.DataParallel(model)
    checkpoint = torch.load(model_ckpt)
    model.load_state_dict(checkpoint)
    model.eval()
    image_size = 32

    CIFAR_SIZE = 32
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    CIFAR_TRANSFORM = trans.Compose([
        trans.ToTensor()])

    testset = dset.CIFAR10(root = data_root, train=False, download=True, transform=CIFAR_TRANSFORM)
    attacker = SimBA(model, 'cifar', image_size)

    # load sampled images or sample new ones
    # this is to ensure all attacks are run on the same set of correctly classified images
    batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
    if os.path.isfile(batchfile):
        checkpoint = torch.load(batchfile)
        images = checkpoint['images']
        labels = checkpoint['labels']
    else:
        images = torch.zeros(args.num_runs, 3, image_size, image_size)
        labels = torch.zeros(args.num_runs).long()
        preds = labels + 1
        while preds.ne(labels).sum() > 0:
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
            for i in list(idx):
                images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
            preds[idx], _ = get_preds(model, images[idx], 'cifar', batch_size=args.batch_size)
        torch.save({'images': images, 'labels': labels}, batchfile)

    if args.order == 'rand':
        n_dims = 3 * args.freq_dims * args.freq_dims
    else:
        n_dims = 3 * image_size * image_size
    if args.num_iters > 0:
        max_iters = int(min(n_dims, args.num_iters))
    else:
        max_iters = int(n_dims)
    N = int(math.floor(float(args.num_runs) / float(args.batch_size)))

    for i in range(1):
        images_batch = images[i]
        labels_batch = labels[i]
        # replace true label with random target labels in case of targeted attack
        if args.targeted:
            labels_targeted = labels_batch.clone()
            while labels_targeted.eq(labels_batch).sum() > 0:
                labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
            labels_batch = labels_targeted
        adv = attacker.simba_single(images_batch, labels_batch)

