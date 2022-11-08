
Introduction to Image Attack with Examples
============
In this section, we introduce the usage of image attacks provided in DeepRobust.

.. contents::
    :local: 

White Box Attack for Image Classification
------------
DeepRobust provides the following white box attack algorithms:

- :class:`deeprobust.image.attack.FGSM`
- :class:`deeprobust.image.attack.LBFGS`
- :class:`deeprobust.image.attack.PGD`
- :class:`deeprobust.image.attack.CarliniWagner`

- :class:`deeprobust.image.attack.Onepixel`
- :class:`deeprobust.image.attack.Universal`

Attack Example
------------

    .. code-block:: python
       
       from deeprobust.image.attack.pgd import PGD
       from deeprobust.image.config import attack_params
       from deeprobust.image.utils import download_model
       import torch
       import deeprobust.image.netmodels.resnet as resnet

       URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_50.pt"
       download_model(URL, "$MODEL_PATH$")

       model = resnet.ResNet18().to('cuda')
       model.load_state_dict(torch.load("$MODEL_PATH$"))
       model.eval()

       transform_val = transforms.Compose([transforms.ToTensor()])
       test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                    transform = transform_val),
                    batch_size = 10, shuffle=True)

       x, y = next(iter(test_loader))
       x = x.to('cuda').float()

       adversary = PGD(model, device)
       Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])

