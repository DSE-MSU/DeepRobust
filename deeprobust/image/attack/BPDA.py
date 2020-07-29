"""
https://github.com/lordwarlock/Pytorch-BPDA/blob/master/bpda.py
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

def normalize(image, mean, std):
    return (image - mean)/std

def preprocess(image):
    image = image / 255
    image = np.transpose(image, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = normalize(image, mean, std)
    return image

def image2tensor(image):
    img_t = torch.Tensor(image)
    img_t = img_t.unsqueeze(0)
    img_t.requires_grad_()
    return img_t

def label2tensor(label):
    target = np.array([label])
    target = torch.from_numpy(target).long()
    return target

def get_img_grad_given_label(image, label, model):
    logits = model(image)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, target)
    loss.backward()
    ret = image.grad.clone()
    model.zero_grad()
    image.grad.data.zero_()
    return ret

def get_cw_grad(adv, origin, label, model):
    logits = model(adv)
    ce = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    loss = ce(logits, label) + l2(0, origin - adv) / l2(0, origin)
    loss.backward()
    ret = adv.grad.clone()
    model.zero_grad()
    adv.grad.data.zero_()
    origin.grad.data.zero_()
    return ret

def l2_norm(adv, img):
    adv = adv.detach().numpy()
    img = img.detach().numpy()
    ret = np.sum(np.square(adv - img))/np.sum(np.square(img))
    return ret

def clip_bound(adv):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    adv = adv * std + mean
    adv = np.clip(adv, 0., 1.)
    adv = (adv - mean) / std
    return adv.astype(np.float32)

def identity_transform(x):
    return x.detach().clone()

def BPDA_attack(image,target, model, step_size = 1., iterations = 10, linf=False, transform_func=identity_transform):
    target = label2tensor(target)
    adv = image.detach().numpy()
    adv = torch.from_numpy(adv)
    adv.requires_grad_()
    for _ in range(iterations):
        adv_def = transform_func(adv)
        adv_def.requires_grad_()
        l2 = nn.MSELoss()
        loss = l2(0, adv_def)
        loss.backward()
        g = get_cw_grad(adv_def, image, target, model)
        if linf:
            g = torch.sign(g)
        print(g.numpy().sum())
        adv = adv.detach().numpy() - step_size * g.numpy()
        adv = clip_bound(adv)
        adv = torch.from_numpy(adv)
        adv.requires_grad_()
        if linf:
            print('label', torch.argmax(model(adv)), 'linf', torch.max(torch.abs(adv - image)).detach().numpy())
        else:
            print('label', torch.argmax(model(adv)), 'l2', l2_norm(adv, image))
    return adv.detach().numpy()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import skimage
    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    image = preprocess(skimage.io.imread('test.png'))

    img_t = image2tensor(image)
    BPDA_attack(img_t, 924, resnet18)
    print('L-inf')
    BPDA_attack(img_t, 924, resnet18, step_size = 0.003, linf=True)
